using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML
{
    public class _Root<T, Treq, Tres>
        where Tres : class, new()
        where Treq : class, new()
    {
        /// <summary>
        /// Context for the each Machine Learning
        /// </summary>
        protected MLContext _mlContext;

        /// <param name="resArr">Parsed list of <typeparamref name="T"/> for the training.</param>
        /// <returns>DataView splited for the training.</returns>
        protected IDataView? splitDataView(Treq[] resArr)
        {
            return _mlContext.Data.LoadFromEnumerable(resArr);
        }

        /// <param name="model">Model for the prediction</param>
        /// <param name="target">
        /// Input: <br/>
        /// make sure that you don't fill the output member in that context.
        /// </param>
        /// <returns>Would return a prediction for the <paramref name="target"/></returns>
        protected Tres getPredict(PredictionEngine<Treq, Tres> engine, Treq target)
        {
            return engine.Predict(target);
        }

        /// <summary>
        /// Would save a model in given path as a file.
        /// </summary>
        /// <param name="path">Given model will be saved here</param>
        /// <param name="model">AI Model made</param>
        /// <param name="dataView">DataView splited. Check <c>splitDataView</c> method</param>
        protected void saveModel(string path, ITransformer model, IDataView dataView)
        {
            using (var fs = new FileStream(path, FileMode.Create))
                _mlContext.Model.Save(model, dataView.Schema, fs);
        }

        /// <param name="path">The path where model has been saved.</param>
        /// <returns>Loaded AI model from the file</returns>
        protected ITransformer loadModel(string path)
        {
            ITransformer model;

            using (var stream = new FileStream(path, FileMode.Open))
                model = _mlContext.Model.Load(stream, out var modelSchema);
            return model;
        }

        /// <param name="trainingDataView">Fully processed DataView</param>
        /// <param name="pipeline">Fully processed pipeline for Learning</param>
        /// <returns>Model for the prediction</returns>
        protected ITransformer getModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var _trainedModel = pipeline.Fit(trainingDataView);
            return _trainedModel;
        }
    }

    public class Root<T, Treq, Tres> : _Root<T, Treq, Tres> 
        where Tres : class, new()
        where Treq : class, new()
    {
        public Root(MLContext? mLContext = null)
        {
            if (mLContext == null) _mlContext = new MLContext(); else _mlContext = mLContext;
        }

        /// <summary>
        /// Engine, will actually do predict.
        /// </summary>
        public PredictionEngine<Treq, Tres> engine;

        /// <summary>
        /// Splited dataview
        /// </summary>
        public Treq[] dataView { set { _dataView = splitDataView(value); } }
        protected IDataView? _dataView;

        /// <summary>
        /// AI Model, it would be the object for get result.
        /// </summary>
        protected ITransformer model;
        protected IEstimator<ITransformer> pipeline;

        /// <summary>
        /// Would save a model into zip file.
        /// </summary>
        /// <param name="path"></param>
        public void save(string path = "model.zip")
        {
            saveModel(path, model, _dataView);
        }

        /// <summary>
        /// Would load a model from the zip file.
        /// </summary>
        /// <param name="path"></param>
        public void load(string path = "model.zip")
        {
            model = loadModel(path);
            engine = _mlContext.Model.CreatePredictionEngine<Treq, Tres>(model);
            return;
        }

        /// <summary>
        /// Train the model with given dataView
        /// </summary>
        /// <param name="dataView">
        /// Set this param provided you want to train a model with specified dataView. <br/>
        /// Setting it null would consider that this model will be trained with given dataView(previous set).
        /// </param>
        public void train(IDataView? dataView = null)
        {
            if (dataView == null) model = getModel(_dataView, pipeline);
            else if (_dataView == null) return;
            else model = getModel(dataView, pipeline);

            engine = _mlContext.Model.CreatePredictionEngine<Treq, Tres>(model);
        }
    }
}