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
import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

import com.ansj.vec.domain.Neuron;

import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;
public class transfer {

	private Map<String, Neuron> wordMap = new HashMap<String, Neuron>(); 
	private Map<String, Neuron> pinyinMap = new HashMap<>();
	private Map<String, String> transMap = new HashMap<>();
	
	/**
	 * 训练多少
	 */
	private int layerSize = 100;

	/**
	 * 上下文窗口大小
	 */
	private int window = 5;
	private int negative = 5;

	private int min_reduce = 5;

	private double sample = 1e-3;
	private double alpha = 0.025;
	private double startingAlpha = alpha;
	private double wa = 1;
	private double wb = 1;

	public int EXP_TABLE_SIZE = 1000;
	private int table_size = (int) 1e8;

	private double[] expTable = new double[EXP_TABLE_SIZE];
	private String[] table = new String[table_size];

	private int trainWordsCount = 0;

	private int MAX_EXP = 6;

	private boolean pyfinish = false;
	
	SpanishPronunciation esTransform = new SpanishPronunciation();

	public transfer(Boolean isCbow, Boolean ispinyin, Integer layerSize, Integer window, Double alpha, Double sample) {
		
		if (layerSize != null)
			this.layerSize = layerSize;
		if (window != null)
			this.window = window;
		if (alpha != null)
			this.alpha = alpha;
		if (sample != null)
			this.sample = sample;
	}
	public transfer() {
		
	}

	/**
	 * trainModel
	 * 
	 * @throws IOException
	 */
	private void getTrans() {
		String espronun;
		for (Entry<String, Neuron> word : wordMap.entrySet()) {
			espronun = esTransform.convert_spanish_word_to_phonetic_transcription(word.getKey(), "es_ES");
			if (!pinyinMap.containsKey(espronun)) {
				pinyinMap.put(espronun, new WordNeuron(espronun, 1, layerSize));
			}
		}
	}
	
	
	private void readVocab(File filein,File fileout) throws IOException {
		String pronun;
		int i=0;
		OutputStreamWriter out = new OutputStreamWriter(new FileOutputStream(fileout), "utf-8");
		BufferedWriter write = new BufferedWriter(out);
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filein)))) {
			String temp = null;
			while ((temp = br.readLine()) != null) {
				i++;
				String[] split = temp.split(" ");
				
				for (String string : split) {
					pronun = esTransform.convert_spanish_word_to_phonetic_transcription(string, "es_ES");
					if (pronun == null) {
						pronun = "#unknown#";
					}
					write.write(pronun + " ");
					write.flush();
				}
				write.write("\n");
				write.flush();
				if(i%1000==0)
					System.out.print("第"+i+"行is OK\n");
			}
		}
		
	}
	public void learnFile(File filein,File fileout) throws IOException {
		//getTrans();
		//System.out.print("getTrans is over");
		readVocab(filein,fileout);

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
		File filein=new File("E:\\文本表示语料\\Spanish\\pureeswiki");
		File fileout=new File("E:\\文本表示语料\\Spanish\\pureeswikitransfer");		
		transfer learn = new transfer();
		learn.learnFile(filein, fileout);
		System.out.print("transfer is over");
	}
}

