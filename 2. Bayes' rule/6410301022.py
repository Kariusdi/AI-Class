'''
Assignment  : 2
Date        : 20 Dec 2023
Name        : Chonakan Chumtap 
Student ID  : 6410301022
'''

def userInput():

    prob_Br = int(input("Please fill prob of red box (%) : ")) / 100
    prob_Bb = 1.00 - prob_Br

    apples_Br = int(input("Please fill quantity of \"apples\" in \"Red\" box : "))
    oranges_Br = int(input("Please fill quantity of \"oranges\" in \"Red\" box : "))

    apples_Bb = int(input("Please fill quantity of \"apples\" in \"Blue\" box : "))
    oranges_Bb = int(input("Please fill quantity of \"oranges\" in \"Blue\" box : "))

    Red_box = {
        "prob" : prob_Br,
        "apples" : apples_Br,
        "oranges": oranges_Br
    }

    Blue_box = {
        "prob" : prob_Bb,
        "apples" : apples_Bb,
        "oranges": oranges_Bb
    }

    return Red_box, Blue_box

def BayesRule(Red_Box, Blue_Box):

    # Conditional Probability 
    # P(X|Y) => X means oranges, Y means Red Box
    prob_oranges_Br = Red_Box["oranges"] / (Red_Box["apples"] + Red_Box["oranges"])
    # P(X|Y) => X means oranges, Y means Blue Box
    prob_oranges_Bb = Blue_Box["oranges"] / (Blue_Box["apples"] + Blue_Box["oranges"])
    # P(X|Y) => X means apples, Y means Red Box
    prob_apples_Br = Red_Box["apples"] / (Red_Box["apples"] + Red_Box["oranges"])
    # P(X|Y) => X means apples, Y means Blue Box
    prob_apples_Bb = Blue_Box["apples"] / (Blue_Box["apples"] + Blue_Box["oranges"])

    # Joint Probability of orange and apple
    pxy_joint_oranges_Br = prob_oranges_Br * Red_Box["prob"]
    pxy_joint_oranges_Bb = prob_oranges_Bb * Blue_Box["prob"]
    pxy_joint_apples_Br = prob_apples_Br * Red_Box["prob"]
    pxy_joint_apples_Bb = prob_apples_Bb * Blue_Box["prob"]

    # Sum of P(X|Y)p(Y) => P(X)
    # P(X), X means orange
    prob_orange = round(pxy_joint_oranges_Br + pxy_joint_oranges_Bb, 2)
    # P(X), X means apple
    prob_apple = round(pxy_joint_apples_Br + pxy_joint_apples_Bb, 2)

    # P(Y|X)
    # P(Y|X) => Y means Red Box, X means oranges
    prob_Br_orange = round(pxy_joint_oranges_Br / prob_orange, 2)
    # P(Y|X) => Y means Red Box, X means apples
    prob_Br_apple = round(pxy_joint_apples_Br / prob_apple, 2)
    # P(Y|X) => Y means Blue Box, X means oranges
    prob_Bb_orange = round(pxy_joint_oranges_Bb / prob_orange, 2)
    # P(Y|X) => Y means Blue Box, X means apples
    prob_Bb_apple = round(pxy_joint_apples_Bb / prob_apple, 2)

    px = {"orange": prob_orange, "apple": prob_apple}
    pyx = {"Br_orange": prob_Br_orange, "Br_apple": prob_Br_apple, "Bb_orange": prob_Bb_orange, "Bb_apple": prob_Bb_apple}

    return px, pyx
    

if __name__ == '__main__':


    Red_Box, Blue_Box = userInput()

    print(
    """
    ##### Summery #####
    
    Red box has {} apples and {} oranges
    Blue box has {} apples and {} oranges

    Prob of Red Box     =   {}
    Prob of Blue Box    =   {}
    """.format(Red_Box["apples"], Red_Box["oranges"], 
               Blue_Box["apples"], Blue_Box["oranges"],
               Red_Box["prob"], Blue_Box["prob"])
    )

    px, pyx = BayesRule(Red_Box, Blue_Box)

    print(
    """
    Prob of Orange      =   {}
    Prob of Apple       =   {}
    
    Prob of "orange from red Box"       =   {}
    Prob of "orange from blue Box"      =   {}

    Prob of "apple from red Box"        =   {}
    Prob of "apple from blue Box"       =   {}
          
    """.format(px["orange"], px["apple"],
               pyx["Br_orange"], pyx["Bb_orange"],
               pyx["Br_apple"], pyx["Bb_apple"])
    )