# Configpy


> When develop the server applications, you could encounter a problem for managing the configuration. This issue can be encountered in every place where `configuration management` is needed as well as server applications.

### 1. built-in data structure

#### .1. Constant

- config.py

```python
# config.py
DATABASE_CONFIG = {
    'host': 'localhost',
    'dbname': 'company',
    'user': 'user',
    'password': 'password',
    'port': 3306
}
```

- main.py

```python
import pymysql
import config

def connect_db(dbname):
    if dbname != config.DATABASE_CONFIG['dbname']:
        raise ValueError("Couldn't not find DB with given name")
    conn = pymysql.connect(host=config.DATABASE_CONFIG['host'],
                           user=config.DATABASE_CONFIG['user'],
                           password=config.DATABASE_CONFIG['password'],
                           db=config.DATABASE_CONFIG['dbname'])
    return conn

connect_db('company')
```

#### .2. class

- config.py

```python
class Config:
    APP_NAME = 'myapp'
    SECRET_KEY = 'secret-key-of-myapp'
    ADMIN_NAME = 'administrator'

    AWS_DEFAULT_REGION = 'ap-northeast-2'
    
    STATIC_PREFIX_PATH = 'static'
    ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'gif']
    MAX_IMAGE_SIZE = 5242880 # 5MB
    
class DevelopmentConfig(Config):
    DEBUG = True
    
    AWS_ACCESS_KEY_ID = 'aws-access-key-for-dev'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-dev'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-dev'
    
    DATABASE_URI = 'database-uri-for-dev'
    
class TestConfig(Config):
    DEBUG = True
    TESTING = True
    
    AWS_ACCESS_KEY_ID = 'aws-access-key-for-test'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-test'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-test'
    
    DATABASE_URI = 'database-uri-for-dev'
  

class ProductionConfig(Config):
    DEBUG = False

    AWS_ACCESS_KEY_ID = 'aws-access-key-for-prod'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-prod'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-prod'

    DATABASE_URI = 'database-uri-for-dev'


class CIConfig:
    SERVICE = 'travis-ci'
    HOOK_URL = 'web-hooking-url-from-ci-service'
```

- main.py

```python
import sys
import config

if __name__ == '__main__':
    env = sys.argv[1] if len(sys.argv) > 2 else 'dev'
    
    if env == 'dev':
        app.config = config.DevelopmentConfig
    elif env == 'test':
        app.config = config.TestConfig
    elif env == 'prod':
        app.config = config.ProductionConfig
    else:
        raise ValueError('Invalid environment name')
   
    app.ci = config.CIConfig
```

### 2. external config file

#### .1. config.ini 

- ini

```ini
; config.ini
[DEFAULT]
SECRET_KEY = secret-key-of-myapp
ADMIN_NAME = administrator
AWS_DEFAULT_REGION = ap-northeast-2
MAX_IMAGE_SIZE = 5242880

[TEST]
TEST_TMP_DIR = tests
TEST_TIMEOUT = 20

[CI]
SERVICE = travis-ci
HOOK_URL = web-hooking-url-from-ci-service
```

#### .2. config.json

```json
{
  "DEFAULT": {
    "SECRET_KEY": "secret-key-of-myapp",
    "ADMIN_NAME": "administrator",
    "AWS_DEFAULT_REGION": "ap-northeast-2",
    "MAX_IMAGE_SIZE": 5242880
  },
  "TEST": {
    "TEST_TMP_DIR": "tests",
    "TEST_TIMEOUT": 20
  },
  "CI": {
    "SERVICE": "travis-ci",
    "HOOK_URL": "web-hooking-url-from-ci-service"
  }
}
```

- main.py

```python
# main_with_ini.py
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

secret_key = config['DEFAULT']['SECRET_KEY'] # 'secret-key-of-myapp'
ci_hook_url = config['CI']['HOOK_URL'] # 'web-hooking-url-from-ci-service'

# main_with_json.py
import json

with open('config.json', 'r') as f:
    config = json.load(f)

secret_key = config['DEFAULT']['SECRET_KEY'] # 'secret-key-of-myapp'
ci_hook_url = config['CI']['HOOK_URL'] # 'web-hooking-url-from-ci-service'
```

### 3. environment variable

```python
import os
from myapp import app
secret_key = os.environ.get('SECRET_KEY', None)
if not secret_key:
    raise ValueError('You must have "SECRET_KEY" variable')
app.config['SECRET_KEY'] = secert_key
```

### 4. dynamic loading

> 通过sys.path.append('/opt/settings')  来进行实现

```python
# /opt/settings/config.py
DATABASE_CONFIG = {
    'host': 'localhost',
    'dbname': 'company',
    'user': 'user',
    'password': 'password',
    'port': 3306
}
# main.py
import sys
import pymysql

sys.path.append('/opt/settings')    
import config

def connect_db(dbname):
    if dbname != config.DATABASE_CONFIG['dbname']:
        raise ValueError("Couldn't not find DB with given name")
    conn = pymysql.connect(host=config.DATABASE_CONFIG['host'],
                           user=config.DATABASE_CONFIG['user'],
                           password=config.DATABASE_CONFIG['password'],
                           db=config.DATABASE_CONFIG['dbname'])
    return conn

connect_db('company')
```

### Resource

- https://hackernoon.com/4-ways-to-manage-the-configuration-in-python-4623049e841b



---

> 作者: liudongdong1  
> URL: liudongdong1.github.io/configpy/  

