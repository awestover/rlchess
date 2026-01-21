Grab GPTOSS20b.
Grab some program that is capable of playing chess, ideally at a tunable ELO.
Figure out some way of measuring how good a chess move is. Maybe there's a python package that does this. 
Have gptoss play several chess games against players of various ELOs, with and without giving gptoss CoT.
Record the average quality of gptoss's moves as determined by our score thing. 
Output these final numbers (average quality of move with and without CoT) as a bargraph, and give me some information to contextualize how good these numbers are. 
For instance it'd be helpful to have a rough sense of what ELO these numbers correspond to / imply.
Also make a plot of: 
For all the ELOs that gptoss played against, what fraction of the time gptoss won.