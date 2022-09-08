# Soccer Eye 

__Open Source modular toolkit for soccer footage analytics__

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

# Goals
- Compatible with televised game footage(dynamic)
- Should work with static full field frames + Dynamic occluded frames



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
- [ ] Shooting Hotspots
    - Determining best placements for succesful shotmaking against specific teams. 
    - Based on advancing ideas as worked on [here](https://github.com/danielazevedo/Football-Analytics/blob/master/expected_goals/xG_model_SVM.ipynb)



--------------------

# Requirements 
- OpenCV 
- Scikit-Image 
- PyTorch 
- TensorFlow


# Future 
- Ship as a library rather than scripts 


# Module Demos 
> coming soon !!!

# Challenges 
__Background Variations__<br/>
![background_variation](./assets/background_variation.png)

__Angle Variations__<br/>
![angle_variation](./assets/angle_variation.png)

__Feed Specific Variations__<br/>
![feed_variation](./assets/feed_variation.png)



# Resources, References, Credits. 
- [DevTo/Stephan007](https://dev.to/stephan007/open-source-sports-video-analysis-using-maching-learning-2ag4)
- [PyImageSearch](https://pyimagesearch.com/blog/)
- [Kaggle Bundesliga Dataset](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
- [SoccerNet](https://www.soccer-net.org)
