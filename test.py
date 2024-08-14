import csv
import time
from datetime import timedelta
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class CasualtyExtractor:
    def __init__(self, model_name, tokenizer_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def probe_info(self, text):
        prompt1 = """Extract casualty statistics from tweets.

[Tweet]: #Moscow Russian earthquake death toll at 6,954, over 400k injured &amp; 1.2m people displaced this weekend. https://t.co/o4JC4HqNU4
[Query]: |Deaths|Injured|City|Country|Earthquake|
[Key]: |6954|400000|Moscow|Russia|yes|

###

[Tweet]: Crushing damage in Peking. Until now, 31 reported deaths and 4000 displaced. #chinahurricane #shaking #beijing #china https://t.co/Z3sDfjEHjC4
[Query]: |Deaths|Injured|City|Country|Earthquake|
[Key]: |31|none|Beijing|China|no|

###

[Tweet]: Injuries in massive #Anchorage flood jumps to 724: govt https://t.co/7asdfasdfaDh
[Query]: |Deaths|Injured|City|Country|Earthquake|
[Key]: |none|724|Anchorage|USA|no|

###

[Tweet]: Sudden earthquake in Saudi Arabia 30,000 injured and 4090 killed. #saudi #earthquake https://t.co/6BJNYBN38
[Query]: |Deaths|Injured|City|Country|Earthquake|
[Key]: |4090|30000|none|Saudi Arabia|yes|

###

[Tweet]: BREAKING: Earthquake of 5.9 magnitude in Nice, France, killing 600 and 4,000 injured. #NICE #quake
[Query]: |Deaths|Injured|City|Country|Earthquake|
[Key]: |600|4000|Nice|France|yes|

###

[Tweet]:
"""
        prompt2 = """[Query]: |Deaths|Injuries|City|Country|Earthquake|
[Key]:"""

        
        full_prompt = prompt1 + " " + text + "\n" + prompt2
        iids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
        print("tokenizing finished")
        
        # Increase max_new_tokens or max_length to ensure sufficient generation length
        generated_ids = self.model.generate(iids, do_sample=False, temperature=1.0, max_new_tokens=25, return_dict_in_generate=True, output_scores=True)
        print("generation finished")
        
        generated_text = self.gen_tokenizer.decode(generated_ids.sequences[0])
        
        # Extract the relevant part after the prompt
        return generated_text.split(prompt2)[-1].strip()

def process_csv(input_file, output_file, model_name, tokenizer_name):
    start_time = time.time()
    extractor = CasualtyExtractor(model_name=model_name, tokenizer_name=tokenizer_name)
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        
        rows = []
        for row in reader:
            if not row:  # Skip empty rows
                continue
            
            tweet_text = row[0]  # Assuming tweet text is in the first column
            response = extractor.probe_info(tweet_text)
            # Parsing the response to extract death and injuries (simplified for illustration)
            response_parts = response.split("|")
            pred_death = response_parts[1].strip() if len(response_parts) > 1 else "none"
            pred_injuries = response_parts[2].strip() if len(response_parts) > 2 else "none"
            row.append(pred_death)
            row.append(pred_injuries)
            rows.append(row)

    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Processing time: {str(timedelta(seconds=elapsed_time))} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract casualty statistics from tweets using an LLM.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument('--model_name', type=str, required=True, help="Name or path of the model to use.")
    parser.add_argument('--tokenizer_name', type=str, default=None, help="Name or path of the tokenizer to use. If not provided, defaults to the model name.")

    args = parser.parse_args()

    # If tokenizer_name is not provided, use model_name
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name

    process_csv(
        input_file=args.input_file,
        output_file=args.output_file,
        model_name=args.model_name,
        tokenizer_name=tokenizer_name
    )
