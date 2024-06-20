#!/bin/bash

years=(2015 2016)

# List of DA types
BERT_da_types=("BERT-nouns_adverbs" )
CBERT_da_types=("CBERT-nouns" "CBERT-nouns_adverbs" )
BERTprepend_da_types=("BERT_prepend-nouns" "BERT_prepend-nouns_adverbs")
BERTexpand_da_types=("BERT_expand-nouns" "BERT_expand-nouns_adverbs")

Total_DA=("${BERT_da_types[@]}$" "${CBERT_da_types[@]}" "${BERTprepend_da_types[@]}" "${BERTexpand_da_types[@]}")

base_command="python prepare_bert.py"

# Loop through each year and each DA type to run the command
for year in "${years[@]}"
do
  for da_type in "${BERT_da_types[@]}"
  do
    command="$base_command --year $year --da_type $da_type"
    echo "Running command: $command"
    $command
  done
done