### Observation planning with Robo-AO

Here are a few general points about observation planning:

- For every observation, there must be a star brighter than ~15.5-16 Vmag in 
the 36"x36" FoV, which could be the target itself. LGS AO is not sensitive to 
image motion, so we correct for that in software using a natural guide star.

- Please refer to Figure 13 from Jensen-Clem+ 2017 (submitted to ApJ) 
to get a feeling of the contrast that a 90-second observation in our longpass 
(>600 nm), and Solan i filters would yield. Figure 10 from the same paper gives 
the Strehl ratio as a function of the seeing. Note that the median seeing at the 
KPNO 2.1-m is around 1.3". 

- Please format your target list following the example in the attached file and send 
that to Dima Duev. Please make sure to have the header with column names in it.
We have a longpass filter (>600 nm; code: FILTER_LONGPASS_600), 
and Sloan g, r, i, and z (code: FILTER_SLOAN_X, X c [G, R, I, Z]). You get the 
best Strehl/SNR ratio (for a given exposure time) with an i'-observation. 
The system is not really optimized for g', so I wouldn't recommend using that 
unless it is absolutely necessary. The (primary star) magnitudes listed in the 
target list are used to set up the camera gain; we prefer Vmag's, but other bands 
would work too.
Note that our queue "intelligently" selects targets based on a number of criteria, 
so there is no predefined order in which the targets will get observed.

- You will be able to access your processed data at Robo-AO's 
[archive](http://roboao.caltech.edu/archive). 
Contact Dima Duev for login/password. Please refer to the paper for a description 
of the pipelines that we run on the raw data.


### Example target list
```
name _RAJ2000 _DEJ2000 epoch mag exposure_time_1 filter_code_1 exposure_time_2 filter_code_2 exposure_time_3 filter_code_3 comment 
TargetName1 12:40:40.79 33:36:17.00 2000 11.6 60 FILTER_LONGPASS_600 60 FILTER_SLOAN_I 60 FILTER_SLOAN_Z "Comment_1_without_blank_spaces"
TargetName2 19:41:09.97 33:55:32.50 2000 10.7 60 FILTER_LONGPASS_600 60 FILTER_SLOAN_I 60 FILTER_SLOAN_Z "Comment_2_without_blank_spaces"
```