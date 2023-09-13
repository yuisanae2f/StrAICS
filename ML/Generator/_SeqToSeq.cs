using Microsoft.ML;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using yuisanae2f.CharCraftableCS.Korean;

namespace yuisanae2f.StrAICS.ML.Generator
{
    public class _SeqToSeq
    {
        private static void shred(string datum, List<Request<int>>[] _, Mutex mtx)
        {
            
            List<Request<int>>[] __ = new List<Request<int>>[3] { new List<Request<int>>(), new List<Request<int>>(), new List<Request<int>>() };

            #region A Line
            StringHandler target = new StringHandler("");
            foreach (char c in datum)
            {
                CharHandler str = new CharHandler(c);
                string t = "";
                __[0].Add(new Request<int> { input = target.shredded, output = str.upperVowel });

                t += str.upperVowel;
                __[1].Add(new Request<int> { input = target.shredded + t, output = str.vowel });

                t += str.vowel;
                __[2].Add(new Request<int> { input = target.shredded + t, output = str.underVowel });

                target.value.Append(c);
                target.value += c;

                Console.Write(c);
            }
            #endregion
            mtx.WaitOne();
            _[0].AddRange(__[0]);
            _[1].AddRange(__[1]);
            _[2].AddRange(__[2]);
            mtx.ReleaseMutex();
            Console.WriteLine($"\n!{datum.Length}");
            return;
        }

        protected Mutex mtx = new Mutex();

        protected static List<Request<int>>[] split(string data, int tdCount, Mutex mtx, string splitters = " \n\t,.!?\'\"")
        {

            #region Initialising Request for craftable Lettter(in this case, Korean)
            List<Request<int>>[] _ = {
            new List<Request<int>>(),
            new List<Request<int>>(),
            new List<Request<int>>()
        };
            #endregion

            List<Task> acts = new List<Task>();

            #region A Line
            // acts.Add(Task.Run(() => { shred(data, _, mtx); })); 
            foreach (char splitter in splitters) 
            { 
                foreach (string datum in data.Split(splitter)) 
                {
                    string __ = datum;
                    if( __.Length != 0 && data != __) acts.Add(Task.Run(() => { shred(__, _, mtx); })); 
                } 
            }

            foreach(Task t in acts) t.Wait();
            #endregion

            return _;
        }
    }
}