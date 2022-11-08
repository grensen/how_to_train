 for more info httpsgithub.comgrensengood_vs_bad_code
 based on httpsgithub.comgrensenmulti-core
#if DEBUG
System.Console.WriteLine(Debug mode is on, switch to Release mode);
#endif 
System.Actionstring print = System.Console.WriteLine;

print(nBegin how to train demon);

 get data
AutoData d = new(@Cmnist);

 define neural network 
int[] network = { 784, 100, 100, 10 };
var LEARNINGRATE = 0.001f;
var MOMENTUM = 0.0f;
var EPOCHS = 300;
var BATCHSIZE = 400;
var FACTOR = 0.97f;
var DROP = 0.5f;

RunDemo(network, d, LEARNINGRATE, MOMENTUM, DROP, FACTOR, EPOCHS, BATCHSIZE);

print(nEnd how to train demo);

+---------------------------------------------------------------------+

static void RunDemo(int[] network, AutoData d, float LEARNINGRATE, float MOMENTUM, float DROP, float FACTOR, int EPOCHS, int BATCHSIZE)
{
    System.Console.WriteLine(NETWORK      =  + string.Join(-, network));
    System.Console.WriteLine(LEARNINGRATE =  + LEARNINGRATE);
    System.Console.WriteLine(MOMENTUM     =  + MOMENTUM);
    System.Console.WriteLine(BATCHSIZE    =  + BATCHSIZE);
    System.Console.WriteLine(EPOCHS       =  + EPOCHS +  (50  60,000 = 3,000,000 exampels));
    System.Console.WriteLine(FACTOR       =  + FACTOR + );
    System.Console.WriteLine(DROP         =  + DROP +  (input sample dropout start 50%));

    int sourceSeed = 1337;

    var mGoodTime = 0.0f;
    Net goodMultiNet = new(network);
    System.Console.WriteLine(nStart non reproducible Multi-Core training);
    mGoodTime = RunNet(true, d, goodMultiNet, 60000, LEARNINGRATE, MOMENTUM, DROP, 1, sourceSeed, EPOCHS, BATCHSIZE);
    
    var sGoodTime = 0.0f;
    Net goodSingleNet = new(network);
    System.Console.WriteLine(nStart reproducible Single-Core training);
    sGoodTime = RunNet(false, d, goodMultiNet, 60000, LEARNINGRATE  2, MOMENTUM, DROP, FACTOR, sourceSeed + sourceSeed  2, EPOCHS  10, BATCHSIZE);
    
    System.Console.WriteLine(nTotal time =  + (sGoodTime + mGoodTime).ToString(F2) + s);
}

static float RunNet(bool multiCore, AutoData d, Net neural, int len, float lr, float mom, float drop, float FACTOR, int sourceSeed, int EPOCHS, int BATCHSIZE)
{
    DateTime elapsed = DateTime.Now;
    RunTraining(multiCore, elapsed, d, neural, len, lr, mom, drop, sourceSeed, FACTOR, EPOCHS, BATCHSIZE);
    return RunTest(multiCore, elapsed, d, neural, 10000);

    static void RunTraining(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len, float lr, float mom, float drop, int sourceSeed, float FACTOR, int EPOCHS, int BATCHSIZE)
    {
        float[] delta = new float[neural.weights.Length];
        for (int epoch = 0, B = len  BATCHSIZE, s = sourceSeed; epoch  EPOCHS; epoch++, lr = FACTOR, mom = FACTOR, drop = FACTOR)
        {
            bool[] c = new bool[B  BATCHSIZE];
            for (int b = 0; b  B; b++)
            {

                if (multiCore)
                    System.Threading.Tasks.Parallel.ForEach(
                        System.Collections.Concurrent.Partitioner.Create(b  BATCHSIZE, (b + 1)  BATCHSIZE), range =
                        {
                            for (int x = range.Item1, X = range.Item2; x  X; x++)
                                c[x] = EvalAndTrainAndDrop(x, d.samplesTraining, neural, delta, d.labelsTraining[x], drop, s++);
                        });
                else
                    for (int x = b  BATCHSIZE, X = (b + 1)  BATCHSIZE; x  X; x++)
                        c[x] = EvalAndTrainAndDrop(x, d.samplesTraining, neural, delta, d.labelsTraining[x], drop, s++);

                Update(neural.weights, delta, lr, mom);
            }
            int correct = c.Count(n = n);  for (int i = 0; i  len; i++) if (c[i]) correct++;
            if ((epoch + 1) % (EPOCHS  10) == 0)
                PrintInfo(Epoch =  + (1 + epoch), correct, B  BATCHSIZE, elapsed);
        }
    }
    static float RunTest(bool multiCore, DateTime elapsed, AutoData d, Net neural, int len)
    {
        bool[] c = new bool[len];
        if (multiCore)
            System.Threading.Tasks.Parallel.ForEach(System.Collections.Concurrent.Partitioner.Create(0, len), range =
            {
                for (int x = range.Item1; x  range.Item2; x++)
                    c[x] = EvalTest(x, d.samplesTest, neural, d.labelsTest[x]);
            });
        else
            for (int x = 0; x  len; x++)
                c[x] = EvalTest(x, d.samplesTest, neural, d.labelsTest[x]);

        int correct = c.Count(n = n);  int correct = 0; for (int i = 0; i  c.Length; i++) if (c[i]) correct++;
        PrintInfo(Test, correct, 10000, elapsed);
        return (float)((DateTime.Now - elapsed).TotalMilliseconds  1000.0f);
    }
}

static int EvalAndDrop(int x, Net neural, byte[] samples, Spanfloat neuron, float drop, int seed)
{
    static int FastRand(ref int seed) { return ((seed = (214013  seed + 2531011))  16) & 0x7FFF; }  [0, 32768)
     FeedInput(x, samples, neuron);
    for (int i = 0, ii = x  784; i  784; i += 1)
    {
        var n = samples[ii++];
        if (n  0) 
            if (FastRand(ref seed)  32767.0f  0.5f) 
                neuron[i] = n  255f;
    }
    
    for (int i = 0, ii = x  784; i  784; i += 8)
    {
        var n = samples[ii++];
        if (n  0) neuron[i] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 1] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 2] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 3] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 4] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 5] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 6] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 7] = n  255f;
    }
    
    FeedForward(neuron, neural);
    Softmax(neuron, neural.net[neural.layers]);
    return Argmax(neural, neuron);
}

static void FeedInput(int x, byte[] samples, Spanfloat neuron)
{
    for (int i = 0, ii = x  784; i  784; i += 8)
    {
        var n = samples[ii++];
        if (n  0) neuron[i] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 1] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 2] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 3] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 4] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 5] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 6] = n  255f;
        n = samples[ii++];
        if (n  0) neuron[i + 7] = n  255f;
    }
}

static bool EvalAndTrainAndDrop(int x, byte[] samples, Net neural, float[] delta, byte t, float drop, int seed)
{
    Spanfloat neuron = stackalloc float[neural.neuronLen];

    static int FastRand(ref int seed) { return ((seed = (214013  seed + 2531011))  16) & 0x7FFF; }  [0, 32768)
     FeedInput(x, samples, neuron);
    {
        float prop = drop  32767.0f;
        for (int i = 0, ii = x  784; i  784; i += 1)
        {
            var n = samples[ii++];
            if (n  0)
                if (FastRand(ref seed)  prop)
                    neuron[i] = n  255f;
        }
    }
     int p = Eval(x, neural, samples, neuron);
    FeedInput(x, samples, neuron);
    FeedForward(neuron, neural);
    Softmax(neuron, neural.net[neural.layers]);
    int p = Argmax(neural, neuron);
    if (neuron[neural.neuronLen - neural.net[^1] + t]  0.99)
        BP(neural, neuron, t, delta);
    return p == t;

    static void BP(Net neural, Spanfloat neuron, int target, float[] delta)
    {
        Spanfloat gradient = stackalloc float[neuron.Length];

         output error gradients, hard target as 1 for its class
        for (int r = neuron.Length - neural.net[neural.layers], p = 0; r  neuron.Length; r++, p++)
            gradient[r] = target == p  1 - neuron[r]  -neuron[r];
        for (int i = neural.layers - 1, j = neuron.Length - neural.net[neural.layers], k = neuron.Length, m = neural.weights.Length; i = 0; i--)
        {
            int right = neural.net[i + 1], left = neural.net[i];
            k -= right; j -= left; m -= right  left;

            for (int l = j, w = m; l  left + j; l++, w += right)
            {
                var n = neuron[l];
                if (n  0)
                {
                    int r = 0; var sum = 0.0f;
                    for (; r  right - 8; r += 8)  8
                    {
                        int kr = k + r, wr = w + r;
                        var g = gradient[kr]; delta[wr] += n  g; sum += neural.weights[wr]  g;
                        g = gradient[kr + 1]; delta[wr + 1] += n  g; sum += neural.weights[wr + 1]  g;
                        g = gradient[kr + 2]; delta[wr + 2] += n  g; sum += neural.weights[wr + 2]  g;
                        g = gradient[kr + 3]; delta[wr + 3] += n  g; sum += neural.weights[wr + 3]  g;
                        g = gradient[kr + 4]; delta[wr + 4] += n  g; sum += neural.weights[wr + 4]  g;
                        g = gradient[kr + 5]; delta[wr + 5] += n  g; sum += neural.weights[wr + 5]  g;
                        g = gradient[kr + 6]; delta[wr + 6] += n  g; sum += neural.weights[wr + 6]  g;
                        g = gradient[kr + 7]; delta[wr + 7] += n  g; sum += neural.weights[wr + 7]  g;
                    }
                    for (; r  right; r++)
                    {
                        int wr = r + w;
                        var g = gradient[k + r];
                        sum += neural.weights[wr]  g; delta[wr] += n  g;
                    }
                    gradient[l] = sum;
                }
            }
        }
    }
}

static bool EvalAndTrain(int x, byte[] samples, Net neural, float[] delta, byte t)
{
    Spanfloat neuron = stackalloc float[neural.neuronLen];
    int p = Eval(x, neural, samples, neuron);
    if (neuron[neural.neuronLen - neural.net[^1] + t]  0.99)
        BP(neural, neuron, t, delta);
    return p == t;

    static void BP(Net neural, Spanfloat neuron, int target, float[] delta)
    {
        Spanfloat gradient = stackalloc float[neuron.Length];

         output error gradients, hard target as 1 for its class
        for (int r = neuron.Length - neural.net[neural.layers], p = 0; r  neuron.Length; r++, p++)
            gradient[r] = target == p  1 - neuron[r]  -neuron[r];
        for (int i = neural.layers - 1, j = neuron.Length - neural.net[neural.layers], k = neuron.Length, m = neural.weights.Length; i = 0; i--)
        {
            int right = neural.net[i + 1], left = neural.net[i];
            k -= right; j -= left; m -= right  left;

            for (int l = j, w = m; l  left + j; l++, w += right)
            {
                var n = neuron[l];
                if (n  0)
                {
                    int r = 0; var sum = 0.0f;
                    for (; r  right - 8; r += 8)  8
                    {
                        int kr = k + r, wr = w + r;
                        var g = gradient[kr]; delta[wr] += n  g; sum += neural.weights[wr]  g;
                        g = gradient[kr + 1]; delta[wr + 1] += n  g; sum += neural.weights[wr + 1]  g;
                        g = gradient[kr + 2]; delta[wr + 2] += n  g; sum += neural.weights[wr + 2]  g;
                        g = gradient[kr + 3]; delta[wr + 3] += n  g; sum += neural.weights[wr + 3]  g;
                        g = gradient[kr + 4]; delta[wr + 4] += n  g; sum += neural.weights[wr + 4]  g;
                        g = gradient[kr + 5]; delta[wr + 5] += n  g; sum += neural.weights[wr + 5]  g;
                        g = gradient[kr + 6]; delta[wr + 6] += n  g; sum += neural.weights[wr + 6]  g;
                        g = gradient[kr + 7]; delta[wr + 7] += n  g; sum += neural.weights[wr + 7]  g;
                    }
                    for (; r  right; r++)
                    {
                        int wr = r + w;
                        var g = gradient[k + r];
                        sum += neural.weights[wr]  g; delta[wr] += n  g;
                    }
                    gradient[l] = sum;
                }
            }
        }
    }
}


static bool EvalTest(int x, byte[] samples, Net neural, byte t)
{
    Spanfloat neuron = stackalloc float[neural.neuronLen];
    int p = Eval(x, neural, samples, neuron);
    return p == t;
}
static void FeedForward(Spanfloat neuron, Net neural)
{
    for (int k = neural.net[0], w = 0, i = 0; i  neural.layers; i++)
    {
        int right = neural.net[i + 1];
        for (int l = k - neural.net[i]; l  k; l++, w += right)
        {
            float n = neuron[l];
            if (n = 0) continue;
            int r = 0;
            for (; r  right - 8; r += 8)
            {
                int wr = w + r, kr = k + r;
                float p = neural.weights[wr]  n; neuron[kr] += p;
                p = neural.weights[wr + 1]  n; neuron[kr + 1] += p;
                p = neural.weights[wr + 2]  n; neuron[kr + 2] += p;
                p = neural.weights[wr + 3]  n; neuron[kr + 3] += p;
                p = neural.weights[wr + 4]  n; neuron[kr + 4] += p;
                p = neural.weights[wr + 5]  n; neuron[kr + 5] += p;
                p = neural.weights[wr + 6]  n; neuron[kr + 6] += p;
                p = neural.weights[wr + 7]  n; neuron[kr + 7] += p;
            }
             source loop for (; r  right; r++) { float p = neural.weights[r + w]  n; neuron[r + k] += p; }
            for (; r  right; r++) { float p = neural.weights[r + w]  n; neuron[r + k] += p; }
        }
        k += right;
    }
}
static void Softmax(Spanfloat neuron, int output)
{
    float scale = 0;
    for (int n = neuron.Length - output; n  neuron.Length; n++)
        scale += neuron[n] = MathF.Exp(neuron[n]);
    for (int n = neuron.Length - output; n  neuron.Length; n++)
        neuron[n] = scale;
}
static int Argmax(Net neural, Spanfloat neuron)
{
    float max = neuron[neuron.Length - neural.net[neural.layers]];
    int prediction = 0;
    for (int i = 1; i  neural.net[neural.layers]; i++)
    {
        float n = neuron[i + neuron.Length - neural.net[neural.layers]];
        if (n  max) { max = n; prediction = i; }  grab maxout prediction here
    }
    return prediction;
}
static int Eval(int x, Net neural, byte[] samples, Spanfloat neuron)
{
    FeedInput(x, samples, neuron);
    FeedForward(neuron, neural);
    Softmax(neuron, neural.net[neural.layers]);
    return Argmax(neural, neuron);

}
static void Update(float[] weight, float[] delta, float lr, float mom)
{   
    for (int w = 0; w  weight.Length; w++)
    {
        var d = delta[w]  lr;
        weight[w] += d;
        delta[w] = mom;
    }
}
static void PrintInfo(string str, int correct, int all, DateTime elapsed)
{
    System.Console.WriteLine(str +  accuracy =  + (correct  100.0  all).ToString(F2)
        +  correct =  + correct +  + all +  time =  + ((DateTime.Now - elapsed).TotalMilliseconds  1000.0).ToString(F2) + s);
}

struct Net
{
    public int[] net;
    public int neuronLen, layers;
    public float[] weights;
    public Net(int[] net)
    {
        this.net = net;
        this.neuronLen = net.Sum();
        this.layers = net.Length - 1;
        this.weights = Glorot(this.net);
        static float[] Glorot(int[] net)
        {
            int len = 0;
            for (int n = 0; n  net.Length - 1; n++)
                len += net[n]  net[n + 1];

            float[] weights = new float[len];
            Erratic rnd = new(1337);

            for (int i = 0, w = 0; i  net.Length - 1; i++, w += net[i - 0]  net[i - 1])  layer
            {
                float sd = MathF.Sqrt(6.0f  (net[i] + net[i + 1]));
                for (int m = w; m  w + net[i]  net[i + 1]; m++)  weights
                    weights[m] = rnd.NextFloat(-sd  1.0f, sd  1.0f);
            }
            return weights;
        }
    }
}
struct AutoData  httpsgithub.comgrenseneasy_regression#autodata
{
    public string source;
    public byte[] samplesTest, labelsTest;
    public byte[] samplesTraining, labelsTraining;
    public AutoData(string yourPath)
    {
        this.source = yourPath;

         hardcoded urls from my github
        string trainDataUrl = httpsgithub.comgrensengif_testrawmasterMNIST_Datatrain-images.idx3-ubyte;
        string trainLabelUrl = httpsgithub.comgrensengif_testrawmasterMNIST_Datatrain-labels.idx1-ubyte;
        string testDataUrl = httpsgithub.comgrensengif_testrawmasterMNIST_Datat10k-images.idx3-ubyte;
        string testnLabelUrl = httpsgithub.comgrensengif_testrawmasterMNIST_Datat10k-labels.idx1-ubyte;

         change easy names 
        string d1 = @trainData, d2 = @trainLabel, d3 = @testData, d4 = @testLabel;

        if (!File.Exists(yourPath + d1)
             !File.Exists(yourPath + d2)
               !File.Exists(yourPath + d3)
                 !File.Exists(yourPath + d4))
        {
            System.Console.WriteLine(Data does not exist);
            if (!Directory.Exists(yourPath)) Directory.CreateDirectory(yourPath);

             padding bits data = 16, labels = 8
            System.Console.WriteLine(Download MNIST dataset from GitHub);
            this.samplesTraining = (new System.Net.WebClient().DownloadData(trainDataUrl)).Skip(16).Take(60000  784).ToArray();
            this.labelsTraining = (new System.Net.WebClient().DownloadData(trainLabelUrl)).Skip(8).Take(60000).ToArray();
            this.samplesTest = (new System.Net.WebClient().DownloadData(testDataUrl)).Skip(16).Take(10000  784).ToArray();
            this.labelsTest = (new System.Net.WebClient().DownloadData(testnLabelUrl)).Skip(8).Take(10000).ToArray();

            System.Console.WriteLine(Save cleaned MNIST data into folder  + yourPath + n);
            File.WriteAllBytes(yourPath + d1, this.samplesTraining);
            File.WriteAllBytes(yourPath + d2, this.labelsTraining);
            File.WriteAllBytes(yourPath + d3, this.samplesTest);
            File.WriteAllBytes(yourPath + d4, this.labelsTest); return;
        }
         data on the system, just load from yourPath
        System.Console.WriteLine(Load MNIST data and labels from  + yourPath + n);
        this.samplesTraining = File.ReadAllBytes(yourPath + d1).Take(60000  784).ToArray();
        this.labelsTraining = File.ReadAllBytes(yourPath + d2).Take(60000).ToArray();
        this.samplesTest = File.ReadAllBytes(yourPath + d3).Take(10000  784).ToArray();
        this.labelsTest = File.ReadAllBytes(yourPath + d4).Take(10000).ToArray();
    }
}
class Erratic  httpsjamesmccaffrey.wordpress.com20190520a-pseudo-pseudo-random-number-generator
{
    private float seed;
    public Erratic(float seed2)
    {
        this.seed = this.seed + 0.5f + seed2;   avoid 0
    }
    public float Next()
    {
        var x = Math.Sin(this.seed)  1000;
        var result = (float)(x - Math.Floor(x));   [0.0,1.0)
        this.seed = result;   for next call
        return this.seed;
    }
    public float NextFloat(float lo, float hi)
    {
        var x = this.Next();
        return (hi - lo)  x + lo;
    }
};

