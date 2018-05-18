"""
Dynamic programming
"""
def build_scoring_matrix(alphabet,diag_score,off_diag_score,dash_score):
    """
    :param alphabet: set(['A','T','C','G'])
    :param diag_score: 6
    :param off_diag_score: 2
    :param dash_score: -4
    :return: dictionary of dictionaries
    """
    scoring_matrix = {}
    lst = list(alphabet)
    lst.append('-')
    for row_char in lst:
        scoring_matrix[row_char] = {}
        for col_char in lst:
            if row_char == '-' or col_char == '-':
                scoring_matrix[row_char][col_char] = dash_score
            elif row_char == col_char:
                scoring_matrix[row_char][col_char] = diag_score
            else:
                scoring_matrix[row_char][col_char] = off_diag_score
    return scoring_matrix

def compute_alignment_matrix(seq_x,seq_y,scoring_matrix,global_flag):
    """
    build matrix (top left corner is (0, 0)
    :param seq_x:
    :param seq_y:
    :param scoring_matrix:
    :param global_flag: if true, assign actual calculated value; if false, assign actual positive values or zero for negtive values
    :return: a matrix, top-left corner is (0, 0)
    """
    m_len, n_len = len(seq_x), len(seq_y)
    s_alignment_matrix = [[0 for _ in range(n_len+1)] for _ in range(m_len+1)]
    # print s_alignment_matrix
    for i_idx in range(1, m_len+1):
        last_score = s_alignment_matrix[i_idx-1][0] + scoring_matrix[seq_x[i_idx-1]]['-']
        s_alignment_matrix[i_idx][0] = \
            (global_flag) and last_score or max(0, last_score)

    for j_idx in range(1,n_len+1):
        last_score = s_alignment_matrix[0][j_idx-1] + scoring_matrix['-'][seq_y[j_idx-1]]
        s_alignment_matrix[0][j_idx] = (global_flag) and last_score or max(0, last_score)

    for i_idx in range(1, m_len+1):
        for j_idx in range(1, n_len+1):
            diag_score = s_alignment_matrix[i_idx-1][j_idx-1] + scoring_matrix[seq_x[i_idx-1]][seq_y[j_idx-1]]
            up_score = s_alignment_matrix[i_idx-1][j_idx] + scoring_matrix[seq_x[i_idx-1]]['-']
            left_score = s_alignment_matrix[i_idx][j_idx-1] + scoring_matrix['-'][seq_y[j_idx-1]]
            max_score = max(diag_score,up_score,left_score)
            s_alignment_matrix[i_idx][j_idx] = (global_flag) and max_score or max(0, max_score)
    return s_alignment_matrix

def compute_global_alignment(seq_x,seq_y,scoring_matrix,alignment_matrix):
    idx_i, idx_j = len(seq_x), len(seq_y)
    ret_score = alignment_matrix[idx_i][idx_j]
    x_ret, y_ret = '', ''
    while idx_i != 0 and idx_j!= 0:
        if alignment_matrix[idx_i][idx_j] == (alignment_matrix[idx_i-1][idx_j-1] +
            scoring_matrix[seq_x[idx_i-1]][seq_y[idx_j-1]]):
            # score from diagnoal cell
            x_ret = (seq_x[idx_i-1]) + x_ret
            y_ret = (seq_y[idx_j-1]) + y_ret
            idx_i -= 1
            idx_j -= 1
        elif alignment_matrix[idx_i][idx_j] == (alignment_matrix[idx_i-1][idx_j] +
            scoring_matrix[seq_x[idx_i-1]]['-']):
            # score from above cell
            x_ret = (seq_x[idx_i - 1]) + x_ret
            y_ret = ('-') + y_ret
            idx_i -= 1
        else:
            # score from left cell
            x_ret = ('-') + x_ret
            y_ret = (seq_y[idx_j - 1]) + y_ret
            idx_j -= 1
    while idx_i != 0:
        # idx_j = 0, move upward along first column
        x_ret = (seq_x[idx_i - 1]) + x_ret
        y_ret = ('-') + y_ret
        idx_i -= 1
    while idx_j != 0:
        # idx_i = 0, move left along first row
        x_ret = ('-') + x_ret
        y_ret = (seq_y[idx_j - 1]) + y_ret
        idx_j -= 1
    return (ret_score, x_ret, y_ret)

seq_x = 'ATG'
seq_y = 'ACG'
score_matrix = build_scoring_matrix(set(['A','C','T','G']), 6,2,-4)
align_matrix = compute_alignment_matrix(seq_x,seq_y,score_matrix,True)
print 'align_matrix', align_matrix
print compute_global_alignment(seq_x,seq_y,score_matrix,align_matrix)