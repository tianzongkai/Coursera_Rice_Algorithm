"""
Provide code and solution for Application 4
"""

DESKTOP = True
import time
import math
import random
import urllib2
import matplotlib.pyplot as plt

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
###############################################


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
    local alignment score: 875 
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

# find_local_align()

def generate_null_distribution(seq_x,seq_y,scoring_matrix,num_trials):
    from collections import defaultdict
    scoring_distribution = defaultdict(int)
    for trial in range(num_trials):
        chars = list(seq_y)
        random.shuffle(chars)
        rand_y = ''.join(chars)
        local_alignment_matrix = student.compute_alignment_matrix(seq_x,rand_y,scoring_matrix,False)
        score = student.compute_local_alignment(seq_x,rand_y,scoring_matrix,local_alignment_matrix)[0]
        scoring_distribution[score] += 1
    return scoring_distribution

def q4():
    start = time.clock()
    score_matrix = read_scoring_matrix(PAM50_URL)
    seq_human = read_protein(HUMAN_EYELESS_URL)
    seq_fly = read_protein(FRUITFLY_EYELESS_URL)
    trials = 1000
    scoring_distribution = generate_null_distribution(seq_human, seq_fly, score_matrix, trials)
    end = time.clock()
    scores = scoring_distribution.keys()
    frequency = scoring_distribution.values()
    freq_norm = [100*float(freq)/trials for freq in frequency]
    print 'Running time: %.2f minutes' % (float(end-start)/60)
    print scores, '\n', freq_norm
    plt.bar(scores, freq_norm)
    plt.ylabel('Normalized Frequency %')
    plt.xlabel('Scores')
    plt.title("Scoring Distribution")
    plt.xticks(range(min(scores),max(scores)+1,2))
    plt.show()
    """
    Scores:
    [37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 82, 83] 
    
    Ratio:
    [0.1, 0.1, 0.3, 1.2, 1.9, 2.8, 3.9, 4.6, 6.1, 6.4, 8.1, 6.7, 7.2, 5.7, 4.8, 5.8, 4.6, 3.9, 3.9, 2.6, 2.9, 2.3, 2.6, 2.0, 0.8, 1.8, 1.2, 0.8, 0.8, 0.8, 1.0, 0.4, 0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
    """
# q4()


def q5():
    scores = [37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 82, 83]
    weights = [0.1, 0.1, 0.3, 1.2, 1.9, 2.8, 3.9, 4.6, 6.1, 6.4, 8.1, 6.7, 7.2, 5.7, 4.8, 5.8, 4.6, 3.9, 3.9, 2.6, 2.9, 2.3, 2.6, 2.0, 0.8, 1.8, 1.2, 0.8, 0.8, 0.8, 1.0, 0.4, 0.2, 0.2, 0.3, 0.3, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
    weight_score = [float(score)*weight/100 for score,weight in zip(scores,weights)]
    mean = sum(weight_score)
    scores_minus_mean = [(float(s)-mean) ** 2 for s in scores]
    weight_score = [float(score)*weight/100 for score,weight in zip(scores_minus_mean,weights)]
    var = sum(weight_score)
    std = math.sqrt(var)
    local_alignment_score = 875
    z_score = (float(local_alignment_score) - mean) / std
    print 'Mean: %.2f\nStandard deviation: %.2f\nz-score: %.2f' %(mean, std, z_score)
    """
    Mean: 52.01
    Standard deviation: 6.91
    z-score: 119.12
    """
q5()

"""
Question 6:
The bell-shape distribution from q4 is obtained by randomly shuffling fly protein sequence 1000 times,
 which means such local alignment scores are obtained by chance. If alignment score of original
 human and fly sequence is also obtain by chance, the score should be more likely to fall 
 under range of [mean-3*std, mean+3*std]. 
 
 However, from calcuation of q5, original human-fly local alignment score is 120 standard
  deviation away from mean which is in far tail region of the bell-shape distribution. 
  It means that the original score of 875 is not by chance.
"""

### Question 7 ###
"""
diag_score = 2
off_diag_score = 1
dash_score = 0
"""

### Question 8 ###
#
def check_spelling(checked_word,dist,word_list):
    diag_score = 2
    off_diag_score = 1
    dash_score = 0
    chars = 'abcdefghijklmnopqrstuvwxyz'
    alphabet = set([char for char in chars])
    len_checkedword = len(checked_word)
    scoring_matrix = student.build_scoring_matrix(alphabet, diag_score, off_diag_score, dash_score)
    similar_word_list = []
    for word in word_list:
        global_alignment_matrix = student.compute_alignment_matrix(checked_word,word,scoring_matrix,True)
        global_alignment_score = student.compute_global_alignment(checked_word,word,scoring_matrix,global_alignment_matrix)[0]
        edit_dist = len_checkedword + len(word) - global_alignment_score
        if edit_dist <= dist:
            similar_word_list.append(word)
    return  similar_word_list

def q8():
    word_list = read_words(WORD_LIST_URL)
    for word, dist in zip(['humble', 'firefly'], [1,2]):
        similar_list = check_spelling(word, dist, word_list)
        print 'similar words to %s is %r' %(word, similar_list)
    """
    similar words to humble is ['bumble', 'fumble', 'humble', 'humbled', 'humbler', 'humbles', 'humbly', 'jumble', 'mumble', 'rumble', 'tumble']
    similar words to firefly is ['direly', 'finely', 'fireclay', 'firefly', 'firmly', 'firstly', 'fixedly', 'freely', 'liefly', 'refly', 'tiredly']
    """
# q8()

