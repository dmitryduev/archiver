### Migrate to updated DB schema

In Robo 3T:
```bash
# rename automated pipeline into bright_star pipeline
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.automated": "pipelined.bright_star" } } )
# move raw fits header up
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.bright_star.fits_header": "fits_header" } } )
# rename faint pipeline into faint_star pipeline
db.getCollection('objects').updateMany( {}, { $rename: { "pipelined.faint": "pipelined.faint_star" } } ) 
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