using Microsoft.ML;
using yuisanae2f.CharCraftableCS.Korean;

namespace yuisanae2f.StrAICS.ML.Generator
{
    public class SequenceToSequence : _SeqToSeq
    {
        private Classifier<int>[] mdl;

        private string _dataView;
        private List<Request<int>>[] _s;

        public int tdCount;

        public string dataView
        {
            get { return _dataView; }
            set
            {
                _dataView = value;
                _s = split(value, tdCount, mtx);
            }
        }

        public List<Request<int>>[] raw
        {
            get { return _s; }
        }

        public void train()
        {
            Thread[] tds = new Thread[3];

            for (int i = 0; i < 3; i++)
            {
                int index = i;
                tds[i] = new Thread(new ThreadStart(() => { mdl[index].dataView = _s[index].ToArray(); mdl[index].train(); }));
            }

            foreach (Thread td in tds) { td.Start(); }
            foreach (Thread td in tds) { td.Join(); }
        }

        public CharHandler getPredictSingle(string root)
        {
            string _root = root;
            StringHandler stringHandler = new StringHandler(_root);

            string rtn = "";
            for (int i = 0; i < 3; i++)
            {
                rtn += (char)(mdl[i].predict(_root).predicted);
            }

            CharHandler chr = new CharHandler('가');
            chr.shredded = rtn;
            return chr;
        }

        public StringHandler getPredict(string root, int row = 1)
        {
            string _root = root;
            StringHandler stringHandler = new StringHandler(_root);

            _root = stringHandler.shredded;
            string __r = "";

            for(int j = 0; j < row; j++)
            {
                for (int i = 0; i < 3; i++)
                {
                    char c = (char)(mdl[i].predict(_root).predicted);
                    __r += c; _root += c;
                }
                stringHandler.shredded = __r;
            }

            return stringHandler;
        }

        public SequenceToSequence(MLContext? up = null, MLContext? vowel = null, MLContext? down = null)
        {
            mdl = new Classifier<int>[]
            {
                new Classifier<int>(up),
                new Classifier<int>(vowel),
                new Classifier<int>(down)
            };

            tdCount = 5;
        }
    }
}