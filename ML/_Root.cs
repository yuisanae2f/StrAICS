using Microsoft.ML;

namespace yuisanae2f.StrAICS.ML
{
    /// <summary>
    /// 이 클래스는 모델을 훈련시키고 사용하는 함수를 제공하는 추상 클래스입니다.
    /// 이 모델은 <typeparamref name="Treq"/>의 형태로 입력을 받아 <typeparamref name="Tres"/>의 형태로 출력합니다 <br/>
    /// </summary>
    /// <typeparam name="T">AI 모델 빌드 후 getPredict를 수행할 시 예상되어 나올 출력의 형태입니다. 어느 자료형이든 간에 상관이 없습니다.</typeparam>
    /// <typeparam name="Treq">
    /// 모델은 이 형태로 된 클래스의 배열로써 모델이 쓸 데이터베이스를 사용하며, 또한 이 클래스를 통해 모델이 사용할 입력을 받습니다. <br/>
    /// 무조건 클래스의 형태를 해야 합니다.
    /// </typeparam>
    /// <typeparam name="Tres">
    /// 모델은 <typeparamref name="Treq"/>의 형태로 입력을 받아 이 형태로 예상을 수행합니다. <br/>
    /// 무조건 클래스의 형태를 해야 합니다.
    /// </typeparam>
    public class _Root<T, Treq, Tres>
        where Tres : class, new()
        where Treq : class, new()
    {
        /// <summary>
        /// <typeparamref name="Treq"/>의 배열을 모델의 학습을 위한 형태로 바꾸는 작업을 수행합니다.
        /// </summary>
        /// <param name="ctx">ML.NET에서 표준으로 사용하는 모델의 학습 환경 정의자</param>
        /// <param name="resArr">학습에 사용될 데이터 묶음</param>
        /// <returns>전처리가 완료되어 <paramref name="ctx"/>에서 정의된 모델 학습에 사용될 수 있는 데이터베이스</returns>
        protected IDataView? splitDataView(MLContext ctx, Treq[] resArr)
        {
            return ctx.Data.LoadFromEnumerable(resArr);
        }

        /// <summary>
        /// 학습이 완료된 모델을 사용하여 예측 작업을 수행한다.
        /// </summary>
        /// <param name="engine">학습이 완료된 모델</param>
        /// <param name="target"><paramref name="engine"/>이 예측 작업을 수행할 때 받을 입력값</param>
        /// <returns>예측 작업을 수행하여 나온 출력값</returns>
        protected Tres getPredict(PredictionEngine<Treq, Tres> engine, Treq target)
        {
            return engine.Predict(target);
        }

        /// <summary>
        /// 학습이 완료된 모델을 학습에 사용된 데이터와 함께 zip 형태의 압축 파일로 <paramref name="path"/>에 저장한다.
        /// </summary>
        /// <param name="ctx">ML.NET에서 표준으로 사용하는 모델의 학습 환경 정의자</param>
        /// <param name="path">학습이 완료된 모델 <paramref name="model"/>이 저장될 경로</param>
        /// <param name="model">학습이 완료된 모델</param>
        /// <param name="dataView">학습에 사용된 데이터베이스</param>
        protected void saveModel(MLContext ctx, string path, ITransformer model, IDataView dataView)
        {
            using (var fs = new FileStream(path, FileMode.Create))
                ctx.Model.Save(model, dataView.Schema, fs);
        }

        /// <summary>
        /// <paramref name="path"/>에 zip의 형태로 저장된 모델을 불러오는 작업을 수행한다.
        /// </summary>
        /// <param name="ctx">ML.NET에서 표준으로 사용하는 모델의 학습 환경 정의자</param>
        /// <param name="path">모델이 저장된 경로</param>
        /// <returns><paramref name="path"/>에서 불러온 모델</returns>
        protected ITransformer loadModel(MLContext ctx, string path)
        {
            ITransformer model;

            using (var stream = new FileStream(path, FileMode.Open))
                model = ctx.Model.Load(stream, out var modelSchema);
            return model;
        }

        /// <summary>
        /// 예측 작업을 수행할 수 있는 모델의 학습을 수행한다.
        /// </summary>
        /// <param name="trainingDataView">모델의 학습에 사용될 데이터베이스</param>
        /// <param name="pipeline"><paramref name="trainingDataView"/>를 사용하여 모델을 학습시키는 방식</param>
        /// <returns>학습이 완료된 모델</returns>
        protected ITransformer getModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var _trainedModel = pipeline.Fit(trainingDataView);
            return _trainedModel;
        }
    }
}