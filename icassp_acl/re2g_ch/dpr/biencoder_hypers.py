from torch_util.hypers_base import HypersBase
import os


class BiEncoderHypers(HypersBase):
    def __init__(self):
        super().__init__()
        # use 'facebook/dpr-ctx_encoder-multiset-base', 'facebook/dpr-question_encoder-multiset-base'
        # or 'facebook/dpr-question_encoder-single-nq-base', 'facebook/dpr-ctx_encoder-single-nq-base'
        self.qry_encoder_name_or_path = 'hfl/chinese-roberta-wwm-ext'
        self.ctx_encoder_name_or_path = 'hfl/chinese-roberta-wwm-ext'
        self.encoder_gpu_train_limit = 8  # max number of instances to encode (-1) to disable gradient checkpointing

        # https://github.com/facebookresearch/DPR#2-biencoderretriever-training-in-single-set-mode
        self.max_grad_norm = 2.0  # default from HypersBase is 1.0
        self.learning_rate = 2e-05  # default from HypersBase is 5e-5
        # self.num_train_epochs = 25  (not setting default here since it is so large)

        self.seq_len_q = 512   # max length for query
        self.seq_len_c = 512  # max length for context
        self.debug_location = ''  # where to save debug info
        self.sample_negative_from_top_k = -1  # always take the first negative if <= 0, otherwise sample a negative from the top-k
        self.disable_confict_free_batches = False  # we check that batches do not contain instances s.t. batch negatives will actually be positives
        self.__required_args__ = []

    def _post_init(self):
        super()._post_init()
        if self.resume_from:
            self.qry_encoder_name_or_path = os.path.join(self.resume_from, 'qry_encoder')
            self.ctx_encoder_name_or_path = os.path.join(self.resume_from, 'ctx_encoder')
