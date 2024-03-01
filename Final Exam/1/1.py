def SpamOrNot(msg):

    setOfmsg = msg.split()
    # print(setOfmsg)
    friends = []
    offers = []
    money = []
    love = []


    for MSG in setOfmsg:
        msglower = MSG.lower()
        if msglower == 'friend' or msglower == 'friends':
            friends.append(msglower)
        elif msglower == 'offer' or msglower == 'offers':
            offers.append(msglower)
        elif msglower == 'money':
            money.append(msglower)
        elif msglower == 'love':
            love.append(msglower)
    
    # print(friends, offers, money, love)

    notSPAM_Worddata = {
        "friend": 61,
        "offer": 32,
        "money": 13,
        "love": 44
    }

    total_notSPAM = sum(notSPAM_Worddata.values())

    SPAM_Worddata = {
        "friend": 14,
        "offer": 43,
        "money": 62,
        "love": 21
    }

    total_SPAM = sum(SPAM_Worddata.values())

    # Conditional Probability 
    # P(X|Y) of all NotSPAM words
    prob_friend_notSPAM = notSPAM_Worddata["friend"] / total_notSPAM
    prob_offer_notSPAM = notSPAM_Worddata["offer"] / total_notSPAM
    prob_money_notSPAM = notSPAM_Worddata["money"] / total_notSPAM
    prob_love_notSPAM = notSPAM_Worddata["love"] / total_notSPAM

    # P(X|Y) of all SPAM words
    prob_friend_SPAM = SPAM_Worddata["friend"] / total_SPAM
    prob_offer_SPAM = SPAM_Worddata["offer"] / total_SPAM
    prob_money_SPAM = SPAM_Worddata["money"] / total_SPAM
    prob_love_SPAM = SPAM_Worddata["love"] / total_SPAM

    ns = total_notSPAM / (total_SPAM + total_notSPAM)
    s = total_SPAM / (total_SPAM + total_notSPAM)

    if friends and offers and money and love:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_offer_notSPAM * prob_money_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_offer_SPAM * prob_money_SPAM * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and offers and money:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_offer_notSPAM * prob_money_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_offer_SPAM * prob_money_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and offers and love:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_offer_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_offer_SPAM * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and offers:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_offer_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_offer_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and money and love:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_money_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_money_notSPAM * prob_love_notSPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and money:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_money_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_money_notSPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends and love:
        prob_notSPAM = ns * prob_friend_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_friend_SPAM * prob_love_notSPAM

        return prob_notSPAM, prob_SPAM
    
    elif friends:
        prob_notSPAM = ns * prob_friend_notSPAM
        prob_SPAM = ns * prob_friend_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif offers and money and love:
        prob_notSPAM = ns * prob_offer_notSPAM * prob_money_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_offer_SPAM * prob_money_SPAM * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif offers and money:
        prob_notSPAM = ns * prob_offer_notSPAM * prob_money_notSPAM
        prob_SPAM = ns * prob_offer_SPAM * prob_money_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif offers and love:
        prob_notSPAM = ns * prob_offer_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_offer_SPAM * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif offers:
        prob_notSPAM = ns * prob_offer_notSPAM
        prob_SPAM = ns * prob_offer_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif money and love:
        prob_notSPAM = ns * prob_money_notSPAM * prob_love_notSPAM
        prob_SPAM = ns * prob_money_SPAM * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif money:
        prob_notSPAM = ns * prob_money_notSPAM
        prob_SPAM = ns * prob_money_SPAM

        return prob_notSPAM, prob_SPAM
    
    elif love:
        prob_notSPAM = ns * prob_love_notSPAM
        prob_SPAM = ns * prob_love_SPAM

        return prob_notSPAM, prob_SPAM
    
    else:
        return "There's no stat words", "There's no stat words"


if __name__ == '__main__':

    msg = str(input("Enter the Message: "))
    notspam, spam = SpamOrNot(msg)
    print(notspam, spam)

    if notspam > spam:
        print("It's probably a NOT SPAM message")
    else:
        print("It's probably a SPAM message")