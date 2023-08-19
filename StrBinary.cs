using Microsoft.ML;
using Microsoft.ML.Data;
using yuisanae2f.StrAICS.ML.Binary.MultiClassification;

namespace yuisanae2f.StrAICS.ML.Binary
{
    public class Response : Response<bool>
    {
        /// <summary>
        /// The probability that given input could be predicted "true"
        /// </summary>
        [ColumnName("Probability")]
        public float probability { get; set; }

        /// <summary>
        /// The probability that the prediction had now isn't wrong
        /// </summary>
        public float score { get { return predicted ? probability : 1 - probability; } }
    }

    public class Binary : Root<bool, Request<bool>, Response>
    {
        /// <summary>
        /// Initialiser for this Object.
        /// </summary>
        /// <param name="mLContext">
        /// Custom mlContext. <br/>
        /// If not given it would generate by itself.
        /// </param>
        public Binary(MLContext? mLContext = null) : base(mLContext)
        {
            pipeline =
                _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "input")
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                    ));
        }

        public Response predict(string target) { return getPredict(engine, new Request<bool> { input = target }); }
    }

    namespace MultiClassification
    {
        public class req
        {
            [LoadColumn(0)] public string input;
            [LoadColumn(1)] public int cond;
            [LoadColumn(2), ColumnName("Label")] public bool output;
        }
    }

    public class Classifier<T> : Root<bool, req, Response>
    {
        public struct res
        {
            public float[] scores;
            public T predicted;
        }

        /// <summary>
        /// Splited dataview
        /// </summary>
        public new Request<T>[] dataView
        {
            set
            {
                _labels = new List<T>();
                List<req> reqs = new List<req>();

                foreach(T output in value.Select(x => x.output))
                {
                    if (!_labels.Contains(output)) _labels.Add(output);
                }

                foreach(Request<T> datum in value)
                {
                    for (int i = 0; i < _labels.Count; i++)
                    {
                        reqs.Add(new req()
                        {
                            input = datum.input,
                            cond = i,
                            output = Equals(_labels[i], datum.output)
                        });
                    }
                }

                _dataView = _mlContext.Data.LoadFromEnumerable(reqs);
            }
        }

        public List<T> labels { get { return _labels; } }
        private List<T> _labels;

        public res predict(string target)
        {
            res _ = new res();

            List<float>  resArr = new List<float>();
            for (int i = 0; i < _labels.ToArray().Length; i++)
            {
                resArr.Add(getPredict(engine, new req { input = target, cond = i}).probability);
            } _.scores = resArr.ToArray();

            _.predicted = _labels[resArr.IndexOf(resArr.Max())];

            return _;
        }

        public Classifier(MLContext? mLContext = null) : base(mLContext)
        {
            pipeline =
                _mlContext.Transforms.Conversion.MapValueToKey("Label")
                .Append(_mlContext.Transforms.Conversion.ConvertType("condFloat", "cond"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("input"))
                .Append(_mlContext.Transforms.Concatenate("Features", "input", "condFloat"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression());
        }
    }

    public class Generator<T> : Classifier<T>
    {
        public Generator(MLContext? mLContext = null) : base(mLContext)
        {

        }
    }
}