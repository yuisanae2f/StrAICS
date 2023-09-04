using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML
{
    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="Treq"></typeparam>
    /// <typeparam name="Tres"></typeparam>
    public class Root<T, Treq, Tres> : _Root<T, Treq, Tres>
        where Tres : class, new()
        where Treq : class, new()
    {
        protected MLContext _mlContext;
        public PredictionEngine<Treq, Tres> engine;

        public Treq[] dataView { set { _dataView = splitDataView(_mlContext, value); } }
        protected IDataView? _dataView;

        protected ITransformer model;
        protected IEstimator<ITransformer> pipeline;

        public Root(MLContext? mLContext = null, Treq[]? dataForTrain = null)
        {
            if (mLContext == null) _mlContext = new MLContext(); else _mlContext = mLContext;
            if (dataForTrain != null) dataView = dataForTrain;
        }

        public void save(string path = "model.zip")
        {
            saveModel(_mlContext, path, model, _dataView);
        }

        public void load(string path = "model.zip")
        {
            model = loadModel(_mlContext, path);
            engine = _mlContext.Model.CreatePredictionEngine<Treq, Tres>(model);
            return;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dataView">너가 이미 전처리를 해 놓은 데이터가 있니</param>
        public void train()
        {
            if (_dataView == null) return;
            model = getModel(_dataView, pipeline);
            engine = _mlContext.Model.CreatePredictionEngine<Treq, Tres>(model);
        }
    }
}

// © 2023. YuiSanae2f