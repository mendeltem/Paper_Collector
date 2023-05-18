# Paper_Collector
This project automates the process of fetching scholarly articles from arXiv using various key phrases. The key phrases are formed by combining a list of relevant keywords. The resulting articles are then saved in .xlsx format.

Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

Prerequisites
To run this project, you will need:

Python 3.6 or later
pandas
resp
os
To install the dependencies, run:

bash
Copy code
pip install pandas resp
Installing
Clone this repository to your local machine.
Ensure you have all the necessary dependencies installed.
Running the Script
You can modify the single_word_key_list in the main function with your own set of keywords.
The output_dir variable should be the path to the directory where you want to save the Excel files.
Run the script using Python 3.
bash
Copy code
python filename.py
Usage
The project will iterate through all combinations of the keywords defined in the single_word_key_list variable, fetching articles from arXiv for each combination.

The results are stored in a pandas DataFrame and then saved as an Excel (.xlsx) file in the output directory specified by output_dir. The files are named in the format "papers_{combination_number}_{file_number}.xlsx", where combination_number is the index of the keyword combination, and file_number is incremented each time the number of articles exceeds the max_dev_temp threshold (default 1000).

Built With
Python
pandas
resp - Used for accessing the arXiv API
