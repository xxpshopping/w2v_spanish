package com.ansj.vec;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Map.Entry;
import com.ansj.vec.util.MapCount;
import com.ansj.vec.util.SpanishPronunciation;
import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;

public class Serial {

	private Map<String, Neuron> wordMap = new HashMap<>();
	private Map<String, Neuron> pinyinMap = new HashMap<>();
	/**
	 * 训练多少个特征
	 */
	private int layerSize = 100;

	/**
	 * 上下文窗口大小
	 */
	private int window = 5;
	private int negative = 5;

	private int min_reduce = 5;

	private double sample = 1e-3;//负采样概率
	private double alpha = 0.025;
	private double startingAlpha = alpha;
	private double wa = 1;
	private double wb = 1;
	
	public int EXP_TABLE_SIZE = 1000;//对f的运算结果进行缓存，需要的时候查表
	private int table_size = (int) 1e8;

	private double[] expTable = new double[EXP_TABLE_SIZE];
	private String[] table = new String[table_size];

	private int trainWordsCount = 0;//训练单词总数

	
	private int MAX_EXP = 6;//最大计算到6 (exp^6 / (exp^6 + 1))，最小计算到-6 (exp^-6 / (exp^-6 + 1))

	private boolean pyfinish = false;
	SpanishPronunciation esTransform = new SpanishPronunciation();

	public Serial(Boolean isCbow, Boolean ispinyin, Integer layerSize, Integer window, Double alpha, Double sample) {
		createExpTable();
		if (layerSize != null)
			this.layerSize = layerSize;
		if (window != null)
			this.window = window;
		if (alpha != null)
			this.alpha = alpha;
		if (sample != null)
			this.sample = sample;
	}

	public Serial() {
		createExpTable();
	}

	/**
	 * trainModel
	 * 
	 * @throws IOException
	 */
	private void trainModel(File file) throws IOException {
		for (int trainNum = 0; trainNum < 2; trainNum++) {
			try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
				String temp = null;
				long nextRandom = 5;
				int wordCount = 0;//目前待训练word总数
				int lastWordCount = 0;//还没训练的word个数
				int wordCountActual = 0;//已经训练完的word个数
				while ((temp = br.readLine()) != null) {//从文件中按行读入
					if (wordCount - lastWordCount > 10000) {
						System.out.println("alpha:" + alpha + "\tProgress: "
								+ (int) (wordCountActual / (double) (trainWordsCount + 1) * 100) + "%");
						wordCountActual += wordCount - lastWordCount;
						lastWordCount = wordCount;
						alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));//自动调整学习速率
						if (alpha < startingAlpha * 0.0001) {
							alpha = startingAlpha * 0.0001;//学习速率有下限
						}
					}
					String[] strs = temp.split(" ");//从文件中读入的每一行以空格分开，存入strs
					wordCount += strs.length;//文件中词的个数
					List<WordNeuron> sentence = new ArrayList<WordNeuron>();
					for (int i = 0; i < strs.length; i++) {
						if (wordMap.containsKey(strs[i])) {
							Neuron entry = wordMap.get(strs[i]);
							if (entry == null) {
								continue;
							}
							// The subsampling randomly discards frequent words
							if (sample > 0) {
								double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
										* (sample * trainWordsCount) / entry.freq;
								nextRandom = nextRandom * 25214903917L + 11;
								if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
									continue;
								}
							}
							sentence.add((WordNeuron) entry);
						}
					}

					for (int index = 0; index < sentence.size(); index++) {
						nextRandom = nextRandom * 25214903917L + 11;
						cbowGram(index, sentence, (int) nextRandom % window);
					}

				}
				br.close();
			}
			if (pyfinish == false) {
				// set word embedding and temp embedding
				String pronun;
				for (Entry<String, Neuron> word : wordMap.entrySet()) {
					WordNeuron wordNeuron = (WordNeuron) word.getValue();
					for (int i = 0; i < layerSize; i++) {
						wordNeuron.tempsyn0[i] = 0;
					}
					pronun = esTransform.convert_spanish_word_to_phonetic_transcription(word.getKey(), "es_ES");
					WordNeuron pyNeuron = (WordNeuron) pinyinMap.get(pronun);
					for (int i = 0; i < layerSize; i++) {
						wordNeuron.tempsyn0[i] += pyNeuron.syn0[i];
					}
				}
			}
			pyfinish = true;
		}
		System.out.println("Vocab size: " + wordMap.size());
		System.out.println("Words in train file: " + trainWordsCount);
		System.out.println("sucess train over!");

	}

	/**
	 * 词袋模型
	 * 
	 * @param index
	 * @param sentence
	 * @param b
	 */
	private void cbowGram(int index, List<WordNeuron> sentence, int b) {
		WordNeuron word = sentence.get(index);
		int a, c = 0;

		double[] neu1e = new double[layerSize]; // 形误差项
		double[] neu1 = new double[layerSize]; // 形表示
		double[] neu2 = new double[layerSize]; // 音表示
		WordNeuron last_word;
		WordNeuron tempy;
		String pronun;

		for (a = b; a < window * 2 + 1 - b; a++)
			if (a != window) {
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;

				if (pyfinish) {
					for (c = 0; c < layerSize; c++) {
						neu1[c] += last_word.syn0[c];
					}
				}

				pronun = esTransform.convert_spanish_word_to_phonetic_transcription(last_word.name, "es_ES");
				tempy = (WordNeuron) pinyinMap.get(pronun);
				for (int i = 0; i < layerSize; i++) {
					neu2[i] = neu2[i] + tempy.syn0[i];
				}

			}

		if (negative > 0) {
			String target;
			int label;
			int next_random;
			Random random = new Random();
			double f1, f2, g, q, f;
			WordNeuron targetNeuron = null;

			for (int d = 0; d < negative + 1; d++) {
				if (d == 0) {
					target = word.name;
					label = 1;
				} else {
					next_random = random.nextInt(table_size);
					target = table[next_random];
					if (target == word.name)
						continue;
					label = 0;
				}
				f1 = 0;
				f2 = 0;

				double[] tempembedding = new double[layerSize];
				if (pyfinish) {
					targetNeuron = (WordNeuron) wordMap.get(target);
					for (int i = 0; i < layerSize; i++) {
						tempembedding[i] = targetNeuron.tempsyn0[i];
					}
				} else {
					pronun = esTransform.convert_spanish_word_to_phonetic_transcription(target, "es_ES");
					tempy = (WordNeuron) pinyinMap.get(pronun);
					for (int i = 0; i < layerSize; i++) {
						tempembedding[i] = tempembedding[i] + tempy.tempsyn0[i];
					}
				}

				for (c = 0; c < layerSize; c++) {
					if (pyfinish) {
						f1 += neu1[c] * tempembedding[c];
					}
					f2 += neu2[c] * tempembedding[c];

				}
				f = f1 + f2;
				if (f > MAX_EXP) {
					q = 1;
				} else if (f < -MAX_EXP) {
					q = 0;
				} else {
					q = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				}

				g = alpha * (label - q);
				for (c = 0; c < layerSize; c++) {
					neu1e[c] += g * tempembedding[c];
				}

				if (pyfinish) {
					for (c = 0; c < layerSize; c++)
						targetNeuron.tempsyn0[c] += g * (neu1[c] + neu2[c]);
				} else {
					pronun = esTransform.convert_spanish_word_to_phonetic_transcription(target, "es_ES");
					tempy = (WordNeuron) pinyinMap.get(pronun);
					for (c = 0; c < layerSize; c++) {
						tempy.tempsyn0[c] += g * neu2[c];
					}

				}
			}
		}

		for (a = b; a < window * 2 + 1 - b; a++) {
			if (a != window) {
				c = index - window + a;
				if (c < 0)
					continue;
				if (c >= sentence.size())
					continue;
				last_word = sentence.get(c);
				if (last_word == null)
					continue;

				if (pyfinish) {
					for (c = 0; c < layerSize; c++)
						last_word.syn0[c] += neu1e[c];
				}

				pronun = esTransform.convert_spanish_word_to_phonetic_transcription(last_word.name, "es_ES");
				tempy = (WordNeuron) pinyinMap.get(pronun);
				for (int i = 0; i < layerSize; i++) {
					tempy.syn0[i] = tempy.syn0[i] + neu1e[i];
				}
			}

		}
	}

	// reduce the vocabulary
	private void ReduceVocab() {
		List<String> r = new ArrayList<String>();
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			if (word.getValue().freq < min_reduce) {
				r.add(word.getKey());
			}
		}
		for (String string : r) {
			wordMap.remove(string);
		}
	}

	/**
	 * 统计词频
	 * 
	 * @param file
	 * @throws IOException
	 */
	private void readVocab(File file) throws IOException {
		MapCount<String> mc = new MapCount<>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)))) {
			String temp = null;
			while ((temp = br.readLine()) != null) {
				String[] split = temp.split(" ");
				trainWordsCount += split.length;
				for (String string : split) {
					if (string.length() > 0 && !string.equals(" ")) {
						mc.add(string);
					}
				}
			}
		}
		for (Entry<String, Integer> element : mc.get().entrySet()) {
			WordNeuron tempw = new WordNeuron(element.getKey(), (double) element.getValue(), layerSize);
			wordMap.put(element.getKey(), tempw);
		}
	}

	private void getPinOrChar() {
		// add the pinyin to the pinyinMap
		String espronun;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			espronun = esTransform.convert_spanish_word_to_phonetic_transcription(word.getKey(), "es_ES");
			if (!pinyinMap.containsKey(espronun)) {
				pinyinMap.put(espronun, new WordNeuron(espronun, 1, layerSize));
			}
		}
	}

	/**
	 * 对文本进行预分类
	 * 
	 * @param files
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	private void readVocabWithSupervised(File[] files) throws IOException {
		for (int category = 0; category < files.length; category++) {
			// 对多个文件学习
			MapCount<String> mc = new MapCount<>();
			try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(files[category])))) {
				String temp = null;
				while ((temp = br.readLine()) != null) {
					String[] split = temp.split(" ");
					trainWordsCount += split.length;
					for (String string : split) {
						mc.add(string);
					}
				}
			}
			for (Entry<String, Integer> element : mc.get().entrySet()) {
				double tarFreq = (double) element.getValue() / mc.size();
				if (wordMap.get(element.getKey()) != null) {
					double srcFreq = wordMap.get(element.getKey()).freq;
					if (srcFreq >= tarFreq) {
						continue;
					} else {
						Neuron wordNeuron = wordMap.get(element.getKey());
						wordNeuron.category = category;
						wordNeuron.freq = tarFreq;
					}
				} else {
					wordMap.put(element.getKey(), new WordNeuron(element.getKey(), tarFreq, category, layerSize));
				}
			}
		}
	}

	/**
	 * Precompute the exp() table f(x) = x / (x + 1)
	 */
	//查表法，用于sigmoid函数的计算
	private void createExpTable() {
		for (int i = 0; i < EXP_TABLE_SIZE; i++) {
			expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
			expTable[i] = expTable[i] / (expTable[i] + 1);
		}
	}

	/**
	 * 构建Negative Sampling用的负采样表
	 */
	private void initUnigramTable() {
		int i;
		long train_words_pow = 0;
		double dl = 0, power = 0.75;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			train_words_pow += Math.pow(word.getValue().freq, power);
		}

		i = 0;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			if (i == 0) {
				dl = Math.pow(word.getValue().freq, power) / train_words_pow;
			} else {
				dl += Math.pow(word.getValue().freq, power) / train_words_pow;
			}
			while (i / (double) table_size < dl && i < table_size) {
				table[i++] = word.getKey();
			}
		}
	}

	/**
	 * 根据文件学习
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void learnFile(File file) throws IOException {
		readVocab(file);
		ReduceVocab();

		if (negative > 0) {
			initUnigramTable();
		}

		getPinOrChar();
		trainModel(file);
	}

	/**
	 * 根据预分类的文件学习
	 * 
	 * @param summaryFile
	 *            合并文件
	 * @param classifiedFiles
	 *            分类文件
	 * @throws IOException
	 */
	public void learnFile(File summaryFile, File[] classifiedFiles) throws IOException {
		readVocabWithSupervised(classifiedFiles);
		new Haffman(layerSize).make(wordMap.values());
		// 查找每个神经元
		for (Neuron neuron : wordMap.values()) {
			((WordNeuron) neuron).makeNeurons();
		}
		trainModel(summaryFile);
	}

	/**
	 * 保存模型
	 */
	public void saveModel(File file) {
		// TODO Auto-generated method stub

		try {
			OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(file), "utf-8");
			BufferedWriter write = new BufferedWriter(out);

			write.write(wa + " " + wb + "\r\n");
			write.flush();
			// write the word vector to file
			write.write("the word vector is \r\n");
			write.flush();

			double[] syn0 = new double[layerSize];
			for (Entry<String, Neuron> element : wordMap.entrySet()) {
				write.write(element.getKey() + " ");
				write.flush();

				for (int i = 0; i < layerSize; i++) {
					syn0[i] = ((WordNeuron) element.getValue()).syn0[i];
				}

				for (int j = 0; j < layerSize; j++) {
					write.write(syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();
			}

			write.write("the temp vector is \r\n");
			write.flush();

			for (Entry<String, Neuron> element : wordMap.entrySet()) {
				write.write(element.getKey() + " ");
				write.flush();

				for (int i = 0; i < layerSize; i++) {
					syn0[i] = ((WordNeuron) element.getValue()).tempsyn0[i];
				}

				for (int j = 0; j < layerSize; j++) {
					write.write(syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();
			}

			// write the pinyin vector to the file
			write.write("the pinyin vector is \r\n");
			write.flush();

			for (Entry<String, Neuron> py : pinyinMap.entrySet()) {
				write.write(py.getKey() + " ");
				write.flush();
				for (int j = 0; j < layerSize; j++) {
					write.write(((WordNeuron) py.getValue()).syn0[j] + " ");
					write.flush();
				}
				write.write("\r\n");
				write.flush();

			}

			wordMap.clear();
			pinyinMap.clear();
			write.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public int getLayerSize() {
		return layerSize;
	}

	public void setLayerSize(int layerSize) {
		this.layerSize = layerSize;
	}

	public int getWindow() {
		return window;
	}

	public void setWindow(int window) {
		this.window = window;
	}

	public double getSample() {
		return sample;
	}

	public void setSample(double sample) {
		this.sample = sample;
	}

	public double getAlpha() {
		return alpha;
	}

	public void setAlpha(double alpha) {
		this.alpha = alpha;
		this.startingAlpha = alpha;
	}

	public static int getParam(String para, String[] args) {
		int i;
		for (i = 0; i < args.length; i++) {
			if (args[i].equals(para)) {
				return i + 1;
			}
		}
		return -1;
	}

	public static void main(String[] args) throws IOException {

		int count = 10, k;
		String inputBaseFile = null;
		String outputBaseFile = null;
		if ((k = getParam("-count", args)) > 0) {
			count = Integer.valueOf(args[k]);
		}
		System.out.println("the count of the train is " + count);

		if ((k = getParam("-input", args)) > 0) {
			inputBaseFile = args[k];
		}
		if ((k = getParam("-output", args)) > 0) {
			outputBaseFile = args[k];
		}
		System.out.println("input base file: " + inputBaseFile);
		System.out.println("output base file: " + outputBaseFile);

		for (int i = 1; i <= count; i++) {
			Serial learn = new Serial();
			int j;
			if ((j = getParam("-negative", args)) > 0) {
				learn.negative = Integer.valueOf(args[j]);
			}
			System.out.println("negative: " + learn.negative);

			long start = System.currentTimeMillis();
			learn.learnFile(new File(inputBaseFile + "pureeswiki"));
			System.out.println("use time " + (System.currentTimeMillis() - start));

			learn.saveModel(new File(outputBaseFile + i + "pycbow.txt"));
		}
	}
}
