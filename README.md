# Archiver

This repository contains code that is used for the [Robo-AO](http://roboao.caltech.edu) 
automated data processing together with the (web-)tools to access the data.  

>Robo-AO is the first automated laser guide star system that is currently installed on the Kitt Peak National Observatory's 2.1 meter telescope in Arizona. 

**archiver.py** is the data processing engine.  
**server\_data\_archive.py** is the web-server for data access.

The architecture is intended to be easily adaptable to the needs of a moderately sized (astronomy) project. 

## System overview
- Distributed architecture 
    * (Mostly) written in OO python 3.6
    * Master process utilizes dask.distributed python module
    * Worker processes can be deployed on single machine or moderately-sized cluster
- MongoDB NoSQL DB for house-keeping
- Multiple data reduction pipelines
    * Bright star pipeline
    * High-contrast pipeline
    * Faint star pipeline
    * Extended object pipeline
    * Astrometric solution
    * Strehl ratio computation
- Interactive data products (web) access
    * Powered by Flask python module
    * Previews for individual objects and (nightly) summaries
    * Data streamed to dynamically rendered templates
    * Production deployment using nginx/supervisord/gunicorn
- Easily extendable and customizable
    * Decision chains for different observations
    *Subclasses + JSON config-files
- [Open source!](https://github.com/dmitryduev/archiver)


For scientific and technical details please refer to 
[Jensen-Clem, Duev, Riddle+ 2017](https://arxiv.org/pdf/1703.08867.pdf) 


## How do I deploy the Archiver?

### Prerequisites
* python libraries
  * flask
  * huey (a forked version with a few tweaks)
  * pymongo
  * image_registration (a forked version with a few tweaks)
  * vip
  * lacosmicx 
  ...

- Install fftw3
On mac:
```
brew install fftw
```
On Fedora:
```
yum install fftw3
```
- Install pyfftw (also see their github page for details) (use the right pip! (the one from anaconda)):
```
pip install pyfftw
```
- Clone `image_registration` repository from https://github.com/dmitryduev/image_registration.git
 I've made it use pyfftw by default, which is significantly faster than the numpy's fft,
 and quite faster (10-20%) than the fftw3 wrapper used in image_registration by default:
```
git clone https://github.com/dmitryduev/image_registration.git
```
- Install it:
```
cd image_registration
python setup.py install --record files.txt
```
If it fails on python3 conda env, run the setup command again.

- To remove:
```
cat files.txt | xargs rm -rf
```

Clone the lacosmicx repository:
```bash
git clone https://github.com/cmccully/lacosmicx.git
```
Install in a manner similar to `image_registration`


Clone the repository:
```bash
git clone https://github.com/dmitryduev/archiver.git
```

Compile the bright star pipeline code:
```bash
cd archive/roboao
# modify Makefile where necessary
make
```

---

### Configuration file (settings and paths)

* config.json
    * Provided as an example
    * modify paths/settings as necessary
---

### Set up and use MongoDB with authentication
Install MongoDB 3.4
(yum on Fedora; homebrew on MacOS)
On Mac OS use ```homebrew```. No need to use root privileges.
```
brew install mongodb
```
On Fedora, you would likely need to do these manipulation under root (```su -```)
 Create a file ```/etc/yum.repos.d/mongodb.repo```, add the following:  
```
[mongodb]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/7/mongodb-org/3.4/x86_64/ 
gpgcheck=0
enabled=1
```
 Install with yum:
```
yum install -y mongodb-org
```

Edit the config file. Config file location:  
```bash
/usr/local/etc/mongod.conf (Mac OS brewed)
/etc/mongod.conf (Linux)
```

Comment out:
```bash
#  bindIp: 127.0.0.1
```
Add: _(this is actually unnecessary)_
```bash
setParameter:
    enableLocalhostAuthBypass: true
```

Create (a new) folder to store the databases:
```bash
mkdir /Users/dmitryduev/web/mongodb/ 
```
In mongod.conf, replace the standard path with the custom one:
```bash
dbpath: /Users/dmitryduev/web/mongodb/
```

**On Mac (on Fedora, will start as a daemon on the next boot)**
Start mongod without authorization requirement:
```bash
mongod --dbpath /Users/dmitryduev/web/mongodb/ 
```

If you're running MongoDB on a NUMA machive 
(connect with the ```mongo``` command and it will tell you if that's the case):
```bash
numactl --interleave=all mongod -f /etc/mongod.conf
```


Connect to mongodb with mongo and create superuser (on Fedora, proceed as root):
```bash
# Create your superuser
$ mongo
> use admin
> db.createUser(
    {
        user: "admin",
        pwd: "yoursecretpassword", 
        roles: [{role: "userAdminAnyDatabase", db: "admin"}]})
> exit 
```
Connect to mongodb (now not necessary as root)
```bash
mongo -u "admin" -p "yoursecretpassword" --authenticationDatabase "admin" 
```
Add user to your database:
```bash
$ mongo
# This will create a databased called 'roboao' if it is not there yet
> use roboao
# Add user to your DB
> db.createUser(
    {
      user: "roboao",
      pwd: "yoursecretpassword",
      roles: ["readWrite"]
    }
)
# Optionally create collections:
> db.createCollection("objects")
> db.createCollection("aux")
> db.createCollection("users")
# this will be later done from python anyways 
```
If you get locked out, start over (on Linux)
```bash
sudo service mongod stop
sudo service mongod start
```
To run the database manually (i.e. not as a service):
```bash
mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```
Connect to database from pymongo:
```python
from pymongo import MongoClient
client = MongoClient('ip_address_or_uri')
db = client.roboao
db.authenticate('roboao', 'yoursecretpassword')
# Check it out (optional):
db['some_collection'].find_one()
```
#### Add admin user for data access on the website

Connect to database from pymongo and do an insertion:
```python
from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import datetime
client = MongoClient('ip_address_or_uri')
# select database 'roboao'
db = client.roboao
db.authenticate('roboao', 'yoursecretpassword')
coll = db['users']
result = coll.insert_one(
        {'_id': 'admin',
         'password': generate_password_hash('robopassword'),
         'programs': 'all',
         'last_modified': datetime.datetime.now()}
)
```

Refer to this [tutorial](https://docs.mongodb.com/manual/tutorial/convert-standalone-to-replica-set/)
to replicate the database.

**Use [Robo 3T](https://robomongo.org) to display/edit DB data!! It's super handy!**  
Useful tip: check [this](https://docs.mongodb.com/manual/tutorial/enable-authentication/) out.

---

### Start the Archiver

start MongoDB (if not running already):
```bash
mongod --auth --dbpath /Users/dmitryduev/web/mongodb/
```

**Run the Archiver!** (preferably, in a _tmux_ session)
```bash
tmux
python archiver.py config.json
```

### Data access via the web-server

Make sure to install python dependencies:
```
git clone https://github.com/pyvirtobs/pyvo.git
cd pyvo && /path/to/python setup.py install
conda install flask-login
```

Run the data access web-server using the pm2 process manager:
```bash
pm2 start server_data_archive.py --interpreter=/path/to/python -- path/to/config.ini
```

#### A short tutorial on how to use the web interface
    TODO
---

## How to work with the database from MongoDB client

Mark all observations as not distributed (this will force):
```bash
db.getCollection('objects').update({}, 
    { $set: 
        {'distributed.status': False,
         'distributed.last_modified': utc_now()}
    }, 
    {multi: true}
)
```

Force faint pipeline on a target:
```bash
db.getCollection('objects').update_one({'_id': '4_351_Yrsa_VIC_lp600_o_20160925_110427.040912'}, 
    { $set: 
        {'pipelined.faint.status.force_redo': True,
         'pipelined.faint.last_modified': utc_now()}
    }
)
```

Change ownership (PI) of a program:
```bash
db.getCollection('objects').update({'science_program.program_id':'4'}, 
    { $set: 
        {'science_program.program_PI': 'asteroids'}
    }, 
    {multi: true}
)
```

Remove psflib data from _aux_ collection in the database:
```
    db.getCollection('aux').update({}, {$unset: {'psf_lib': ''}}, {multi: true})
```

---

## Archive structure
The processed data are structured in the way described below. 
It should be straightforward to restore the database in case of a 'database disaster' 
keeping this structure in mind (in fact, **archiver.py** will take care of that automatically 
once the database is up and running).

##### Science observations + daily summary plots (seeing, Strehl, contrast curves)
File naming and descriptions:
```
/path/to/archive/
├──yyyymmdd/
   ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS/
   │  ├──bright_star/
   │  │  ├──preview/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_full.png
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_cropped.png
   │  │  ├──strehl/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_strehl.txt
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_box.fits
   │  │  ├──pca/
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.png
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.png
   │  │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_contrast_curve.txt
   │  │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_pca.fits
   │  │  ├──20p.fits
   │  │  └──100p.fits
   │  ├──faint_star/
   │  │  ├──preview/
   │  │  │  └──...
   │  │  ├──strehl/
   │  │  │  └──...
   │  │  ├──pca/
   │  │  │  └──...
   │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_summed.fits
   │  ├──extended_object/
   │  │  ├──preview/
   │  │  │  └──...
   │  │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS_deconvolved.fits
   │  └──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.SSSSSS.tar.bz2
   ├──.../
   ├──summary/
   │  ├──psflib/
   │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.png
   │  │  ├──programID_objectName_camera_filter_mark_yyyymmdd_HHMMSS.fits
   │  │  └──...
   │  ├──seeing/
   │  │  ├──yyyymmdd_hhmmss.png
   │  │  ├──...
   │  │  ├──seeing.yyyymmdd.txt
   │  │  └──seeing.yyyymmdd.png
   │  ├──contrast_curve.yyyymmdd.png
   │  └──strehl.yyyymmdd.png
   └──calib/
      ├──flat_c.fits
      ├──flat_lp600.fits
      ├──flat_Sg.fits
      ├──flat_Sr.fits
      ├──flat_Si.fits
      ├──flat_Sz.fits
      ├──dark_0.fits
      ├──dark_6.fits
      ├──dark_7.fits
      ├──dark_8.fits
      ├──dark_9.fits
      └──dark_10.fits
|──.../
└──psf_library.fits
```

---

## Processing flowcharts

## Flowcharts
If you're seeking to understand how things (should) work

![alt text](/doc/pipeline.png)