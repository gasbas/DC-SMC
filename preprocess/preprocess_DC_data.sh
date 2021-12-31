wget --output-document=2017_raw_data.xlsx https://osse.dc.gov/sites/default/files/dc/sites/osse/publication/attachments/2017%20PARCC%20School%20Level%20Math_0.xlsx?accessType=DOWNLOAD
wget --output-document=2018_raw_data.xlsx https://osse.dc.gov/sites/default/files/dc/sites/osse/page_content/attachments/2018%20PARCC%20School%20Level%20Math.xlsx?accessType=DOWNLOAD
wget --output-document=2019_raw_data.xlsx https://osse.dc.gov/sites/default/files/dc/sites/osse/page_content/attachments/2018-19%20PARCC%20School%20Level%20Math%201.15.20.Xlsx?accessType=DOWNLOAD
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c2sul5nD8Ajt-GR7oup0jneTD7R8pfJm' -O replacements.json

python preprocess_DC_data.py