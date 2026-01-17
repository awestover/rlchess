Grab Qwen/Qwen3-8b.
Grab some program that is capable of playing chess, ideally at a tunable ELO.
Figure out some way of measuring how good a chess move is. Maybe there's a python package that does this. 
Have Qwen play several chess games against players of various ELOs, with and without giving Qwen CoT.
Record the average quality of Qwen's moves as determined by our score thing. 
Output these final numbers (average quality of move with and without CoT) as a bargraph, and give me some information to contextualize how good these numbers are. 
For instance it'd be helpful to have a rough sense of what ELO these numbers correspond to / imply.
Also make a plot of: 
For all the ELOs that Qwen played against, what fraction of the time Qwen won.