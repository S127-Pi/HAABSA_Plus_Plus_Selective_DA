#!/bin/bash

years=(2015 2016)

# List of DA types
BERT_da_types=("BERT-nouns" "BERT-adverbs" "BERT-nouns_adverbs" "BERT-aspect" "BERT-aspect_adverbs")
CBERT_da_types=("CBERT-nouns" "CBERT-adverbs" "CBERT-nouns_adverbs" "CBERT-aspect" "CBERT-aspect_adverbs")
BERTprepend_da_types=("BERT_prepend-nouns" "BERT_prepend-adverbs" "BERT_prepend-nouns_adverbs" "BERT_prepend-aspect" "BERT_prepend-aspect_adverbs")
BERTexpand_da_types=("BERT_expand-nouns" "BERT_expand-adverbs" "BERT_expand-nouns_adverbs" "BERT_expand-aspect" "BERT_expand-aspect_adverbs")
Total_DA=("${BERT_da_types[@]}$" "${CBERT_da_types[@]}" "${BERTprepend_da_types[@]}" "${BERTexpand_da_types[@]}")

base_command="python TorchBert.py"

# Loop through each year and each DA type to run the command
for year in "${years[@]}"
do
  for da_type in "${Total_DA[@]}"
  do
    command="$base_command --year $year --da_type $da_type"
    echo "Running command: $command"
    $command
  done
done