# Soccer Eye 

__Open Source toolkit for soccer analytics__


# Head
- Compatible with televised game footage(dynamic)
- Should work with static full field footage 
- Utilized cutting edge machine learning models + tried and tested computer vision algorithms 


# Checkpoints + Feature List 
## Tooling 
- [ ] Field Localization `FL`
    - Locating the field in a frame 
- [ ] Player Identification + Localization `PL`
    - Identifying player by jersy numbers 
    - Localization of players on the field 
- [ ] Ball Identification + Localization `BL`
    - Identifying and localizing soccer ball in the field 

## Analytics 
- [ ] 2D Birds Eye View 
    - Converting the footage to birds eye view with players and ball mapped 
- [ ] Penalty Analyzer 
    - Analyze penalty shootouts 
        - Ball Curvature 
        - Ball Speed 
        - Shooter Pose 
- [ ] Pattern Crunching 
    - Given a few games of a team, pattern and play recognition for any given team and good counter-patterns. 
    - Analysing attack patterns of teams. 
- [ ] Player Analysis 
    - Discovering player patterns (passing left, right etc.) and biases for building counter patterns. 



--------------------

# Requirements 
- OpenCV 
- Scikit-Image 
- PyTorch 
- TensorFlow


# Future 
- Ship as a library rather than scripts 

# Challenges 
__Background Variations__<br/>
![background_variation]('./assets/background_variation.png')

# Resources 
> Submit additional feature requests as PR 
- [DevTo/Stephan007](https://dev.to/stephan007/open-source-sports-video-analysis-using-maching-learning-2ag4)