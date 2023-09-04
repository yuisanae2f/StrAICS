﻿using Microsoft.ML;
using yuisanae2f.StrAICS.ML.Binary.MultiClassification;

namespace yuisanae2f.StrAICS.ML.Binary
{
    public class Classifier<T> : Root<bool, req, Response>
    {
        public struct res
        {
            public float[] scores;
            public T predicted;
        }

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

            List<float> resArr = new List<float>();
            for (int i = 0; i < _labels.ToArray().Length; i++)
            {
                float pro = getPredict(engine, new req { input = target, cond = i }).probability;
                resArr.Add(pro);
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
}

// © 2023. YuiSanae2f