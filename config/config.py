#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:32:31 2023

@author: temuuleu
"""

import os
from dotenv import load_dotenv
from colorama import Fore

load_dotenv(verbose=True)

from config.singleton import Singleton


class Config(metaclass=Singleton):
    
    def __init__(self) -> None:
        self.output_dir                =  os.getenv("output_dir")

        
               
def check_master_file_list_path() -> None:
    """Check if the MASTERFILE_PATH is set in config.py or as an environment variable."""
    cfg = Config()
    if not cfg.output_dir:
        print(
            Fore.RED
            + "Please set your output_dir in .env or as an environment variable."
        )
        exit(1)
