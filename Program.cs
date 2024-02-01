using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NEAT_test
{
    class Program
    {
        static Random rand = new();
        static List<int> innodes = new(), outnodes = new();

        //hyperparameters
        #region
        //crossover & speciation
        const double compatmod = 1.1;
        static double compattresh = 4;
        const double weightdiffcoe = 1, excesscoe = 2;
        const double elitepercent = 0.2;
        //pop
        const double fittresh = 3.95, playtimes = 4;
        const int popsize = 100, maxgen = -1, speciestarget = 10; //leave maxgen as -1 for unlimited
        static int gen = 0;
        //activation
        const int type = 2, outtype = 0; //0 = sigmoid, 1 = tanh, 2 = ReLU
        //mutation
        const bool boundweights = false;
        const double mutatepow = 2.5, condisturbchance = 0.8, conenablechance = 0.1, conaddchance = 0.1, nodeaddchance = 0.01;
        const int tryaddtimes = 3;
        #endregion

        static void Main(string[] args)
        {
            Run(2, 1, true, @"C:\Users\Kiyonn\source\repos\NEAT Template\Example NNs.txt");
        }

        static double Fitness(NN net)
        {
            //XOR test
            int[] tests = { 0, 1, 2, 3 };
            //test each input
            double fitness = 4 * playtimes;
            for (int plays = 0; plays < playtimes; plays++)
            {
                //randomise order
                for (int i = 0; i < tests.Length; i++)
                {
                    int swapi = rand.Next(i, tests.Length), swap = tests[swapi];
                    tests[swapi] = tests[i];
                    tests[i] = swap;
                }
                foreach (int test in tests)
                {
                    int in1 = test & 1, in2 = test >> 1;
                    fitness -= Math.Pow(net.FeedFoward(new List<double> { in1, in2 })[0] - (in1 ^ in2), 2);
                }
            }

            return Math.Max(fitness / playtimes, 0);
        }

        static NN[] Run(int inputs, int outputs, bool verbose, string path)
        {
            //initialise NNs
            NN[] NNs = new NN[popsize];
            if (File.Exists(path))
            {
                NNs = LoadNNs(path);
                Console.WriteLine("Gen: " + gen);
                Console.WriteLine("Average Fitness: ");
                Console.WriteLine("Average Size: ");
                Console.WriteLine("No. of Species: ");
                Console.WriteLine();
                gen++;
            }
            else for (int i = 0; i < popsize; i++) NNs[i] = new(inputs, outputs);

            //evolve
            for (; gen < maxgen || maxgen == -1; gen++)
            {
                //check fitness
                for (int i = 0; i < popsize; i++) NNs[i].fitness = Fitness(NNs[i]);
                //speciate
                List<List<int>> species = new();
                for (int i = 0; i < popsize; i++)
                {
                    if (species.Count == 0) species.Add(new List<int> { i });
                    else
                    {
                        bool existingsp = false;
                        foreach (List<int> s in species)
                        {
                            if (NNs[i].SameSpecies(NNs[s[0]]))
                            {
                                s.Add(i);
                                existingsp = true;
                                break;
                            }
                        }
                        if (!existingsp) species.Add(new List<int> { i });
                    }
                }
                //adjust compattresh
                if (species.Count < speciestarget) compattresh /= compatmod;
                else if (species.Count > speciestarget) compattresh *= compatmod;
                //explicit fitness sharing
                //find average fitness
                double avgfit = 0, maxfit = 0;
                List<double> spavgfits = new();
                foreach (List<int> s in species)
                {
                    double spavgfit = 0;
                    foreach (int i in s)
                    {
                        spavgfit += NNs[i].fitness;
                        maxfit = Math.Max(maxfit, NNs[i].fitness);
                    }
                    avgfit += spavgfit;
                    spavgfits.Add(spavgfit / s.Count);
                }
                avgfit /= popsize;
                //calc amount of babies
                int babies = 0;
                int[] babiesdis = new int[species.Count];
                for (int i = 0; i < babiesdis.Length; i++)
                {
                    babiesdis[i] = (int)Math.Floor(spavgfits[i] / avgfit * species[i].Count);
                    babies += babiesdis[i];
                }
                //if there is space for extra babies
                while (babies < popsize)
                {
                    double max = 0;
                    int maxi = 0;
                    //distribute babies to the fittest species first
                    for (int i = 0; i < spavgfits.Count; i++)
                    {
                        if (spavgfits[i] > max)
                        {
                            max = spavgfits[i];
                            maxi = i;
                        }
                    }
                    spavgfits[maxi] = float.MinValue;
                    babiesdis[maxi]++;
                    babies++;
                }
                //log
                if (verbose)
                {
                    List<NN> save = new(NNs);
                    save.Sort((y, x) => x.fitness.CompareTo(y.fitness));
                    SaveToFile(path.Insert(path.LastIndexOf('.'), " save"), save.ToArray());
                    double avgsize = 0;
                    foreach (NN nn in NNs) avgsize += nn.nodes.Count;
                    avgsize /= popsize;
                    Console.WriteLine("Gen: " + gen);
                    Console.WriteLine("Average Fitness: " + avgfit);
                    Console.WriteLine("Average Size: " + avgsize);
                    Console.WriteLine("No. of Species: " + species.Count);
                    Console.WriteLine();
                }
                //break if tresh reached
                if (maxfit >= fittresh)
                {
                    List<NN> save = new(NNs);
                    save.Sort((y, x) => x.fitness.CompareTo(y.fitness));
                    SaveToFile(path, save.ToArray());
                    break;
                }
                //mating season
                List<NN> newNNs = new();
                foreach (List<int> s in species) s.Sort((y, x) => NNs[x].fitness.CompareTo(NNs[y].fitness)); //sort in order of descending fitness
                for (int i = 0; i < species.Count; i++)
                {
                    //only top n% can reproduce
                    List<int> canrepro = species[i].GetRange(0, (int)Math.Ceiling(species[i].Count * elitepercent));
                    if (babiesdis[i] > 0) newNNs.Add(NNs[canrepro[0]]); //add best from species without altering
                    if (canrepro.Count != 1)
                    {
                        //prepare roulette wheel
                        double[] wheel = new double[canrepro.Count - 1];
                        wheel[0] = NNs[canrepro[0]].fitness;
                        for (int j = 1; j < wheel.Length; j++) wheel[j] = wheel[j - 1] + NNs[canrepro[j]].fitness;
                        //make babeeeeeee
                        for (int j = 0; j < babiesdis[i] - 1; j++)
                        {
                            //spin da wheel
                            int parent1 = rand.Next((int)wheel[wheel.Length - 1]), parent2 = rand.Next((int)wheel[wheel.Length - 1]), index;
                            for (index = 0; wheel[index] < parent1; index++) ;
                            parent1 = species[i][index];
                            for (index = 0; wheel[index] < parent2; index++) ;
                            parent2 = species[i][index];
                            //lucky parents get lovey dovey
                            if (NNs[parent1].fitness > NNs[parent2].fitness) newNNs.Add(NNs[parent2].Mate(NNs[parent1]));
                            else newNNs.Add(NNs[parent1].Mate(NNs[parent2]));
                        }
                    }
                    else for (int j = 0; j < babiesdis[i] - 1; j++) newNNs.Add(NNs[canrepro[0]].Mate(NNs[canrepro[0]])); //selfcest cuz he's lonely
                }
                newNNs.CopyTo(NNs);
                //save progress
                SaveToFile(path, NNs);
            }

            return NNs;
        }

        class NN
        {
            public int inputs, outputs;
            public double fitness = 0;
            public List<Node> nodes = new();
            public List<int> connectionids = new();
            public Dictionary<int, Connection> connections = new();

            public NN(int _inputs, int _outputs, List<Connection> _connections = null)
            {
                inputs = _inputs + 1;
                outputs = _outputs;
                if (_connections == null)
                {
                    //make input nodes
                    for (int i = 0; i < inputs; i++)
                    {
                        //connect every input to every output
                        List<Connection> outs = new List<Connection>();
                        for (int j = 0; j < outputs; j++)
                        {
                            outs.Add(new(i, inputs + j, GaussRand()));
                            connections.Add(outs[j].id, outs[j]);
                            connectionids.Add(outs[j].id);
                        }
                        nodes.Add(new(nodes, i, _outputs: new(outs)));
                    }
                    //make output nodes
                    for (int i = 0; i < outputs; i++)
                    {
                        List<Connection> ins = new List<Connection>();
                        for (int j = 0; j < inputs; j++) ins.Add(connections[outputs * j + i]);
                        nodes.Add(new(nodes, inputs + i, true, _inputs: new(ins)));
                    }
                }
                else
                {
                    foreach (Connection c in _connections)
                    {
                        //add connection to connection tracking lists
                        Connection newc = c.Clone();
                        connectionids.Add(newc.id);
                        connections.Add(newc.id, newc);
                        //add nodes as nescessary
                        while (nodes.Count <= newc.input || nodes.Count <= newc.output) nodes.Add(new(nodes, nodes.Count));
                        //add connection to coresponding nodes
                        nodes[c.input].outputs.Add(newc);
                        nodes[c.output].inputs.Add(newc);
                    }
                }
            }

            public List<double> FeedFoward(List<double> input)
            {
                //bias node
                input.Add(1);
                //set input nodes
                for (int i = 0; i < inputs; i++) nodes[i].value = nodes[i].UpdateValue() + input[i];
                //update all nodes (except output)
                for (int i = inputs + outputs; i < nodes.Count; i++) nodes[i].UpdateValue();
                //update ouput nodes and get output
                List<double> output = new();
                for (int i = inputs; i < inputs + outputs; i++) output.Add(nodes[i].UpdateValue());
                return output;
            }

            public NN Mate(NN fitter)
            {
                //clone
                NN child = fitter.Clone();
                //corss
                foreach (int c in child.connections.Keys)
                {
                    if (connections.ContainsKey(c))
                    {
                        //50% of the time change weights
                        if (rand.NextDouble() < 0.5) child.connections[c].weight = connections[c].weight;
                        //10% of the time make enabled same as less fit one
                        if (rand.NextDouble() < 0.1) child.connections[c].enabled = connections[c].enabled;
                    }
                }
                //mutate
                child.Mutate();
                return child;
            }

            public NN Mutate()
            {
                //mutate connections
                foreach (Connection c in connections.Values)
                {
                    //mutate the weight
                    if (rand.NextDouble() < condisturbchance)
                    {
                        if (rand.NextDouble() < 0.1) c.weight = GaussRand();              //10% chance to completely change weight
                        else if (rand.NextDouble() < 0.5) c.weight += UniformRand() / 25 * mutatepow; //45% chance to slightly change weight
                        else c.weight *= 1 + (UniformRand() / 40 * mutatepow);                        //45% chance to slightly scale weight
                    }
                    //enable/disable connection
                    if (rand.NextDouble() < conenablechance) c.enabled = !c.enabled;
                    //keep weight within bounds
                    if (boundweights) c.weight = c.weight < 0 ? Math.Max(c.weight, -mutatepow * 2) : Math.Min(c.weight, mutatepow * 2);
                }

                //add a connection between existing nodes
                if (rand.NextDouble() < conaddchance)
                {
                    for (int i = 0; i < tryaddtimes; i++) //try twice
                    {
                        int inid = rand.Next(nodes.Count), outid = rand.Next(nodes.Count);
                        if (outid == inid) outid = (outid + 1) % nodes.Count;
                        Connection newcon = new(inid, outid, GaussRand());
                        if (!connections.ContainsKey(newcon.id))
                        {
                            nodes[inid].outputs.Add(newcon);
                            nodes[outid].inputs.Add(newcon);
                            connections.Add(newcon.id, newcon);
                            connectionids.Add(newcon.id);
                            break;
                        }
                    }
                }

                //add a node between existing connection
                if (rand.NextDouble() < nodeaddchance)
                {
                    //original connection to "break"
                    Connection breakcon = connections[connectionids[rand.Next(connections.Count)]];
                    for (int i = 0; i < tryaddtimes - 1 && !breakcon.enabled; i++) breakcon = connections[connectionids[rand.Next(connections.Count)]];
                    //disable original connection
                    breakcon.enabled = false;
                    //insert node inbetween
                    Connection incon = new(breakcon.input, nodes.Count, breakcon.weight), outcon = new(nodes.Count, breakcon.output, 1);
                    connections.Add(incon.id, incon);
                    connections.Add(outcon.id, outcon);
                    connectionids.Add(incon.id);
                    connectionids.Add(outcon.id);
                    nodes[breakcon.input].outputs.Add(incon);
                    nodes[breakcon.output].inputs.Add(outcon);
                    nodes.Add(new(nodes, nodes.Count, false, new() { incon }, new() { outcon }));
                }

                return this;
            }

            public bool SameSpecies(NN other)
            {
                int matching = 0, largegenomenorm = Math.Max(1, Math.Max(connections.Count, other.connections.Count) - 20);
                double weightdiff = 0;
                //go through each connection and see if it is excess or matching
                foreach (int conid in connections.Keys)
                {
                    if (other.connections.ContainsKey(conid))
                    {
                        double weight = connections[conid].enabled ? connections[conid].weight : 0, otherweight = other.connections[conid].enabled ? other.connections[conid].weight : 0;
                        weightdiff += Math.Abs(weight - otherweight);
                        matching++;
                    }
                }
                //return whether or not they're the same species
                if (matching == 0) return excesscoe * (connections.Count + other.connections.Count - 2 * matching) / largegenomenorm < compattresh;
                else return (weightdiffcoe * weightdiff / matching) + (excesscoe * (connections.Count + other.connections.Count - 2 * matching) / largegenomenorm) < compattresh;
            }

            public NN Clone()
            {
                return new(inputs - 1, outputs, new(connections.Values));
            }
        }

        class Node
        {
            public bool isoutput;
            public List<Node> network;
            public double value = 0;
            public int id;
            public List<Connection> inputs = new List<Connection>(), outputs = new List<Connection>();

            public Node(List<Node> _network, int _id, bool _isoutput = false, List<Connection> _inputs = null, List<Connection> _outputs = null)
            {
                network = _network;
                id = _id;
                isoutput = _isoutput;
                if (_inputs != null) inputs = _inputs;
                if (_outputs != null) outputs = _outputs;
            }

            public double UpdateValue()
            {
                //sum activations * weights
                value = 0;
                foreach (Connection c in inputs) if (c.enabled) value = Math.FusedMultiplyAdd(network[c.input].value, c.weight, value);
                //squisification
                if (isoutput)
                {
                    if (outtype == 0) value = Sigmoid(value);
                    else if (outtype == 1) value = tanh(value);
                    else if (outtype == 2) value = ReLU(value);
                }
                else if (type == 0) value = Sigmoid(value);
                else if (type == 1) value = tanh(value);
                else if (type == 2) value = ReLU(value);

                return value;
            }
        }

        class Connection
        {
            public bool enabled = true;
            public int input, output, id;
            public double weight;

            public Connection(int _input, int _output, double _weight)
            {
                input = _input;
                output = _output;
                weight = _weight;
                int checkid = Exists(input, output);
                if (checkid == -1)
                {
                    id = innodes.Count;
                    innodes.Add(input);
                    outnodes.Add(output);
                }
                else id = checkid;
            }

            public Connection Clone()
            {
                Connection clone = new(input, output, weight);
                if (!enabled) clone.enabled = false;
                return clone;
            }
        }

        static double GaussRand()
        {
            double output = 0;
            for (int i = 0; i < 6; i++) output += rand.NextDouble();
            return (output - 3) / 3;
        }
        static double UniformRand()
        {
            return (rand.NextDouble() - 0.5) * 2;
        }
        static double Sigmoid(double input)
        {
            return 1 / (1 + Math.Exp(-input * 4.9));
            //return 1 / (1 + Math.Exp(-input));
        }
        static double tanh(double input)
        {
            //1.7159 * tanh(2 / 3 * x)
            return Math.Tanh(input) / 2 + 0.5f;
        }
        static double ReLU(double input)
        {
            return input > 0 ? input : 0;
        }
        static int Exists(int input, int output)
        {
            for (int i = 0; i < innodes.Count; i++) if (innodes[i] == input && outnodes[i] == output) return i;
            return -1;
        }
        static void SaveToFile(string path, NN[] NNs)
        {
            StringBuilder text = new();
            text.AppendLine(gen.ToString());
            text.AppendLine(compattresh.ToString());
            for (int i = 0; i < NNs.Length; i++)
            {
                text.AppendLine(i + " - Fitness: " + NNs[i].fitness);
                text.AppendLine((NNs[i].inputs - 1) + " " + NNs[i].outputs);
                foreach (Connection c in NNs[i].connections.Values) text.AppendLine(c.input + " " + c.output + " " + c.weight + (c.enabled ? "" : " Disabled"));
                text.AppendLine();
            }
            File.WriteAllText(path, text.ToString(), Encoding.UTF8);
            return;
        }
        static NN[] LoadNNs(string path)
        {
            List<string> lines = new(File.ReadAllLines(path, Encoding.UTF8));
            List<List<string>> pars = new();
            gen = Convert.ToInt32(lines[0]);
            compattresh = Convert.ToDouble(lines[1]);
            int oldi = 3;
            //split up NNs
            for (int i = 4; i < lines.Count; i++)
            {
                if (lines[i].Length == 0)
                {
                    pars.Add(lines.GetRange(oldi, i - oldi));
                    i += 2;
                    oldi = i;
                }
            }

            //create NNS
            NN[] NNs = new NN[pars.Count];
            for (int i = 0; i < NNs.Length; i++)
            {
                //no. of inputs and outputs
                string[] inout = pars[i][0].Split(' ');
                //connections
                List<Connection> cons = new();
                for (int j = 1; j < pars[i].Count; j++)
                {
                    string[] con = pars[i][j].Split(' ');
                    Connection newcon = new(Convert.ToInt32(con[0]), Convert.ToInt32(con[1]), Convert.ToDouble(con[2]));
                    if (con.Length == 4) newcon.enabled = false;
                    cons.Add(newcon);
                }
                //create NN

                NNs[i] = new(Convert.ToInt32(inout[0]), Convert.ToInt32(inout[1]), cons);
            }
            return NNs;
        }
    }
}
