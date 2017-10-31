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