import tempfile
import os
import shutil


class Glove:
    def __init__(self, memory=4.0, min_vocab_count=5, dim=50, max_iter=15, num_threads=8, xmax=10, window_sz=5):
        self.memory = memory
        self.min_vocab_count = min_vocab_count
        self.dim = dim
        self.max_iter=max_iter
        self.num_threads = num_threads
        self.xmax=xmax
        self.window_sz = window_sz
        self.embeddings = dict()

    def fit(self, data_gen, tokenize=lambda line: line.split(" ")):
        temp_dir = tempfile.mkdtemp()
        corpus = os.path.join(temp_dir, "corpus")
        with open(corpus, "wt") as fp:
            for line in data_gen:
                fp.write(" ".join([word.replace("\\", "\\\\").replace("_", "\\_").replace(" ", "_") for word in tokenize(line)]) + "\n")
        bash_cmd = """
        cd {build_dir};
        make;
        {build_dir}/vocab_count -min-count {vocab_min_count} -verbose 1 < {input_file} > {input_file}.vocab;
        {build_dir}/cooccur -memory {mem} -vocab-file {input_file}.vocab -verbose 1 -window-size {window_sz} < {input_file} >  {input_file}.coocor;
        {build_dir}/shuffle -memory {mem} -verbose 1 < {input_file}.coocor > {input_file}.shuff;
        {build_dir}/glove -save-file {out_file} -threads {n_threads} -input-file {input_file}.shuff -x-max {x_max} -iter {max_iter} -vector-size {vec_size} -binary 2 -vocab-file {input_file}.vocab -verbose 1;
        """.format(
            build_dir=os.path.join(os.path.abspath(os.path.dirname(__file__)), "build"),
            input_file=corpus,
            window_sz=self.window_sz,
            mem=self.memory,
            n_threads=self.num_threads,
            out_file=os.path.join(temp_dir, "embeddings"),
            x_max=self.xmax,
            max_iter=self.max_iter,
            vec_size=self.dim,
            vocab_min_count=self.min_vocab_count
        )
        os.system(bash_cmd)
        with open(os.path.join(temp_dir, "embeddings.txt"), "rt") as fp:
            for line in fp:
                word, *vec = line.strip().split(" ")
                word = word.replace("_", " ").replace("\\ ", "_")
                vec = [float(x) for x in vec]
                self.embeddings[word] = vec
        self.tokenize = tokenize
        shutil.rmtree(temp_dir)

    def embed(self, sentence):
        tokens = self.tokenize(sentence)
        vecs = [self.embeddings[token] for token in tokens if token in self.embeddings]
        return [sum(feat_col) for feat_col in zip(*vecs)]




if __name__ == "__main__":
    g = Glove(max_iter=1)
    g.fit(open("text8", "rt").readlines())
    print(g.embed("Hello, if this works I can push and then sleep"))