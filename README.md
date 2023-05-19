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


# For the essentials:

First install resp

1. Install the  arxiv

```shell
git clone https://github.com/monk1337/resp
cd resp 
pip install -r requirements.txt && pip install -e .

1. Install Paper Collector

```
Clone this repository to your local machine.
Ensure you have all the necessary dependencies installed.



```bash

git clone https://github.com/mendeltem/Paper_Collector.git
cd Paper_Collector
pip install -r requirements.txt
```
Set up to .bashrc
'''



# For Running Milvus:

Install Milvus Server:

https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-compose-on-ubuntu-20-04

Step 1 — Installing Docker Compose


```bash
sudo curl -L "https://github.com/docker/compose/releases/download/1.29.2/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
'''


Step 2 — Installing Milvus Server

```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker-compose up -d
sudo docker-compose ps

#stop milvus
sudo docker-compose down

'''


Usage
The project will iterate through all combinations of the keywords defined in the single_word_key_list variable, fetching articles from arXiv for each combination.

The results are stored in a pandas DataFrame and then saved as an Excel (.xlsx) file in the output directory specified by output_dir. The files are named in the format "papers_{combination_number}_{file_number}.xlsx", where combination_number is the index of the keyword combination, and file_number is incremented each time the number of articles exceeds the max_dev_temp threshold (default 1000).


Built With
Python


