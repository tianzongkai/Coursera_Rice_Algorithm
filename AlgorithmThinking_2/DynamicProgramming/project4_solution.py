

def build_scoring_matrix(alphabet,diag_score,off_diag_score,dash_score):
    scoring_matrix = {}
    for row_char in alphabet:
        scoring_matrix[row_char] = {}
        for col_char in alphabet:
            if row_char == '-' or col_char == '-': scoring_matrix[row_char][col_char] = dash_score
            elif row_char == col_char: scoring_matrix[row_char][col_char] = diag_score
            else: scoring_matrix[row_char][col_char] = off_diag_score
    return scoring_matrix

scoring_matrix = build_scoring_matrix({'A','U','C','G', '-'}, 10, 4,-6)

def compute_alignment_matrix(seq_x,seq_y,scoring_matrix,global_flag):
    m_len, n_len = len(seq_x), len(seq_y)
    s_alignment_matrix = [[0 for _ in range(len(seq_y))] for _ in range(len(seq_x))]
    for i_idx in range(1, m_len):
        last_score = s_alignment_matrix[i_idx-1][0] + scoring_matrix[seq_x[i_idx-1]]['-']
        s_alignment_matrix[i_idx][0] = \
            global_flag and last_score or max(0, last_score)
    for j_idx in range(1,n_len):
        last_score = s_alignment_matrix[0][j_idx-1] + scoring_matrix['-'][j_idx-1]
        s_alignment_matrix[0][j_idx] = global_flag and last_score or max(0, last_score)
    for i_idx in range(1, m_len+1):
        for j_idx in range(1, n_len+1):
            diag_score = s_alignment_matrix[i_idx-1][j_idx-1] + scoring_matrix[seq_x[i_idx-1]][seq_y[j_idx-1]]
            up_score = s_alignment_matrix[i_idx-1][j_idx] + scoring_matrix[seq_x[i_idx-1]]['-']
            left_score = s_alignment_matrix[i_idx][j_idx-1] + scoring_matrix['-'][seq_y[j_idx-1]]
            max_score = max(diag_score,up_score,left_score)
            s_alignment_matrix[i_idx][j_idx] = global_flag and max_score or max(0, max_score)
    return s_alignment_matrix