"""
Provide code and solution for Application 4
"""

DESKTOP = True

import math
import random
import urllib2

if DESKTOP:
    import matplotlib.pyplot as plt
    import alg_project4_solution as student
else:
    import simpleplot
    import userXX_XXXXXXX as student

# URLs for data files
PAM50_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_PAM50.txt"
HUMAN_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_HumanEyelessProtein.txt"
FRUITFLY_EYELESS_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_FruitflyEyelessProtein.txt"
CONSENSUS_PAX_URL = "http://storage.googleapis.com/codeskulptor-alg/alg_ConsensusPAXDomain.txt"
WORD_LIST_URL = "http://storage.googleapis.com/codeskulptor-assets/assets_scrabble_words3.txt"


###############################################
# provided code

def read_scoring_matrix(filename):
    """
    Read a scoring matrix from the file named filename.

    Argument:
    filename -- name of file containing a scoring matrix

    Returns:
    A dictionary of dictionaries mapping X and Y characters to scores
    """
    scoring_dict = {}
    scoring_file = urllib2.urlopen(filename)
    ykeys = scoring_file.readline()
    ykeychars = ykeys.split()
    for line in scoring_file.readlines():
        vals = line.split()
        xkey = vals.pop(0)
        scoring_dict[xkey] = {}
        for ykey, val in zip(ykeychars, vals):
            scoring_dict[xkey][ykey] = int(val)
    return scoring_dict


def read_protein(filename):
    """
    Read a protein sequence from the file named filename.

    Arguments:
    filename -- name of file containing a protein sequence

    Returns:
    A string representing the protein
    """
    protein_file = urllib2.urlopen(filename)
    protein_seq = protein_file.read()
    protein_seq = protein_seq.rstrip()
    return protein_seq


def read_words(filename):
    """
    Load word list from the file named filename.

    Returns a list of strings.
    """
    # load assets
    word_file = urllib2.urlopen(filename)

    # read in files as string
    words = word_file.read()

    # template lines and solution lines list of line string
    word_list = words.split('\n')
    print "Loaded a dictionary with", len(word_list), "words"
    return word_list

### Question 1 ###
def find_local_align():
    score_matrix = read_scoring_matrix(PAM50_URL)
    seq_human = read_protein(HUMAN_EYELESS_URL)
    seq_fly = read_protein(FRUITFLY_EYELESS_URL)
    local_alignment_matrix = student.compute_alignment_matrix(seq_human,seq_fly,score_matrix,False)
    score, seq_loc_human, seq_loc_fly = student.compute_local_alignment(seq_human,seq_fly,score_matrix,local_alignment_matrix)
    length = len(seq_loc_fly)
    agree = 0
    for idx in range(length):
        if seq_loc_fly[idx] == seq_loc_human[idx]:
            agree += 1
    print 'Question 1:\n'
    print 'score:', score, '\nhuman:', seq_loc_human, '\nfly:  ', seq_loc_fly
    print 'Agree percentage: %.2f' % (100*float(agree)/length)
    """
    Question 1:
    score: 875 
    human: HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEK-QQ 
    fly:   HSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATAEVVSKISQYKRECPSIFAWEIRDRLLQENVCTNDNIPSVSSINRVLRNLAAQKEQQ
    Agree percentage: 93.98%
    """

    ### Question 2 ###
    print '\nQuestion 2:\n'
    seq_loc_human = seq_loc_human.replace('-','')
    seq_loc_fly = seq_loc_fly.replace('-','')
    seq_pax = read_protein(CONSENSUS_PAX_URL)  #Q2
    # seq_pax = 'ACBEDGFIHKMLNQPSRTWVYXZ' #Q3
    for idx in range(2):
        if idx == 0:
            seq = seq_loc_human
            type = 'human'
        else:
            seq = seq_loc_fly
            type = 'fly'
        global_alignment_matrix = student.compute_alignment_matrix(seq,seq_pax,score_matrix,True)
        score, x_glbl, pax_glbl = student.compute_global_alignment(seq,seq_pax,score_matrix,global_alignment_matrix)
        length = len(x_glbl)
        agree = 0
        for idx in range(length):
            if x_glbl[idx] == pax_glbl[idx]:
                agree += 1

        print 'score:', score, '\n'+type, x_glbl, '\nPAX:  ', pax_glbl
        print type+' agree percentage: %.2f' % (100 * float(agree) / length)
        """
        Question 2:

        human score: 613 
        human: -HSGVNQLGGVFVNGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATPEVVSKIAQYKRECPSIFAWEIRDRLLSEGVCTNDNIPSVSSINRVLRNLASEKQQ 
        PAX:   GHGGVNQLGGVFVNGRPLPDVVRQRIVELAHQGVRPCDISRQLRVSHGCVSKILGRYYETGSIKPGVIGGSKPKVATPKVVEKIAEYKRQNPTMFAWEIRDRLLAERVCDNDTVPSVSSINRIIR--------
        human agree percentage: 72.93
        
        flyscore: 586 
        fly:  -HSGVNQLGGVFVGGRPLPDSTRQKIVELAHSGARPCDISRILQVSNGCVSKILGRYYETGSIRPRAIGGSKPRVATAEVVSKISQYKRECPSIFAWEIRDRLLQENVCTNDNIPSVSSINRVLRNLAAQKEQQ 
        PAX:  GHGGVNQLGGVFVNGRPLPDVVRQRIVELAHQGVRPCDISRQLRVSHGCVSKILGRYYETGSIKPGVIGGSKPKVATPKVVEKIAEYKRQNPTMFAWEIRDRLLAERVCDNDTVPSVSSINRIIR---------
        fly agree percentage: 70.15
        """

        """
        Question 3:
        
        The level of similarity of Q1 and Q2 is not due to chance, but due to mutation
         from the same ancestor over long time.
        
        For two random sequences of similar length, the agreement of alighment wouldn't be
        as high as Q1 and Q2 becasue random sequences have very low chance that they're 
        mutated from the same sequence. 
        """


find_local_align()



