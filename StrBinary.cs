using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Newtonsoft.Json;
using System.Reflection.Metadata.Ecma335;

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

    public class Binary : Root<bool, Response>
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
    }

    public class Classifier<T> : Root<bool, Response>
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
            get { return dataView; } 
            set 
            {
                _labels = new List<T>();

                List<Request<bool>> reqs = new List<Request<bool>>();

                foreach(T output in value.Select(x => x.output))
                {
                    if (!_labels.Contains(output)) _labels.Add(output);
                }

                foreach(Request<T> datum in value)
                {
                    for (int i = 0; i < _labels.Count; i++)
                    {
                        reqs.Add(new Request<bool>()
                        {
                            input = JsonConvert.SerializeObject(new Request<int>() { input = datum.input, output = i }),
                            output = Equals(_labels[i], datum.output)
                        });
                    }
                }

                base.dataView = reqs.ToArray();
            }
        }

        public List<T> labels { get { return _labels; } }
        private List<T> _labels;

        public new res predict(string target)
        {
            res _ = new res();

            List<float>  resArr = new List<float>();
            foreach(T lbl in _labels)
            {
                resArr.Add(base.predict(JsonConvert.SerializeObject(new Request<T>() { input = target, output = lbl })).probability);
            } _.scores = resArr.ToArray();

            _.predicted = _labels[resArr.IndexOf(resArr.Max())];

            return _;
        }

        public Classifier(MLContext? mLContext = null) : base(mLContext)
        {
            pipeline =
                _mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: "input")
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                    labelColumnName: "Label",
                    featureColumnName: "Features"
                    ));
        }
    }
}