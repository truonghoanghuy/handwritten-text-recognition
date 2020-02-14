# Handwritten Text Recognition

### Dataset: VNOnDB
Original data as InkML format can be downloaded from
http://tc11.cvc.uab.es/datasets/HANDS-VNOnDB2018_1 \
Data then can be converted to PNG format using the tool in
https://github.com/vndee/vnondb-extractor \
The result in PNG format can also be downloaded there.

1. Extract the `VNOnDB_processed.zip`. Rename the `Data_processed` to `data`
2. Extract the 3 files inside `VNOnDB_ICFHR2018_dataSplit.zip` into `data/`

Before starting, the data folder will look like this
```text
data
  ├── InkData_line_processed
  |     ├── 20140603_0003_BCCTC_tg_0_0.png
  |     ├── 20140603_0003_BCCTC_tg_0_0.txt
  |     └── ...
  ├── InkData_paragraph_processed
  |     └── ...
  ├── InkData_word_processed
  |     └── ...
  ├── test_set.txt
  ├── train_set.txt
  └── validation_set.txt
```
Then split data into train, validation and test:

        python split_data.py

Then ...
