import numpy as np

# AA_list is the numeric encoding of different AA
# each index corresponds to one of the 20 natrual AA
AA_list = ['A', 'R', 'N', 'D', 'C',
     'Q', 'E', 'G', 'H', 'I',
     'L', 'K', 'M', 'F', 'P',
     'S', 'T', 'W', 'Y', 'V']
AAmap = np.array(AA_list)

# Used to apply the mapping
AA_dict = dict(zip(AA_list, range(20)))

def aa_to_ord(aa):
     '''
     Converts a given AA character representation 
     into its ordinal value

     Parameters
     ----------
     aa: Character
          Character representation of an AA

     Returns
     -------
     int
          Ordinal representation of a given AA
     '''

     try:
          return AA_dict[aa]
     except:
          return -1 # if the AA is not one of the 20

def aa_seq_to_ord(aa_seq):
     '''
     Applies the aa_to_ord function to all elements of
     a given sequence of AA

     Parameters
     ----------
     aa: np.Array
          Array of AA abbriviations

     Returns
     -------
     np.Array
          Array of ordinal representations of the given AAs
     '''
     return np.vectorize(aa_to_ord)(aa_seq)
