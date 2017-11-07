### Migrate to updated DB schema

In Robo 3T:
```bash
# rename automated pipeline into bright_star pipeline
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.automated": "pipelined.bright_star" } } )
# move raw fits header up
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.bright_star.fits_header": "fits_header" } } )
# rename faint pipeline into faint_star pipeline
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.faint": "pipelined.faint_star" } } ) 
# put in additional fields for bright_star pipeline:
db.getCollection('objects').updateMany( {"pipelined.bright_star": {$exists: 1}, "pipelined.bright_star.status.enqueued": {$exists: 0}}, { $set: { "pipelined.bright_star.status.enqueued": false, "pipelined.bright_star.status.force_redo": false, "pipelined.bright_star.status.retries": 1 } } )
# change raw field
db.getCollection('objects').updateMany( {"raw_data.location": {$size: 1}}, { $set: { "raw_data.location": ["140.252.53.120:22220", "/nas_data/VIC_data/"] } } )
``` 

#### Analysis machine
Fix config.json!

Create folders:
```bash
mkdir /Data1/archive
mkdir /Data1/_tmp 
```

Copy compiled Nick's pipeline
```bash
cp /home/roboao/web/archiver/roboao /home/roboao/web/archiver/roboao
```

Install anaconda3 or create new python environment:
```bash
conda create --name python3 python=3.6
source activate python3
```