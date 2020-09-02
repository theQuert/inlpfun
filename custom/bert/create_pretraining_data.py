import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "Input raw file.")

flags.DEFINE_string("output_file", None, "Ouput TF example file.")

flags.DEFINE_string("vocab_file", None, "The vocab file that the BERT model was trained on.")

flags.DEFINE_bool("do_lower_case", True, "Whether to lower case the input text. Should be true for uncased "
"models and False for cased models.")

flags.DEFINE_bool("do_whole_word_mask", False, "Whether to use whole word masking rather than per-WordPiece"
"masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length")

flags.DEFINE_integer("max_prediction_per_seq", 20, "Maximum number of masked LM predictions per sequence")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation")

flags.DEFINE_integer("dupe_factor", 10, "Number of times to duplicate the input data (with different masks)." )

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability")

flags.DEFINE_float(
    "short_seq_prob", 0.1, 
    "Probability of creating sequences which are shorter than the 'maximum length'"
)

class TrainingInstance(object):
    '''A single training instance (sentence pair)'''
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = " " 
        s += "token: %s\n" % (" ".join( 
             [tokenization.printable_text(x) for x in self.token]))
        s += "segment_ids: %s\n" %(" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" %(" ".join()
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += 

        
          

