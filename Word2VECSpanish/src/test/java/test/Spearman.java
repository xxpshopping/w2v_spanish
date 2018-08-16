package test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import com.ansj.vec.util.SpanishPronunciation;

public class Spearman {
	ArrayList<ArrayList<String>> simword = new ArrayList<ArrayList<String>>();
	ArrayList<Double> trainwordsim = new ArrayList<Double>();
	ArrayList<Integer> loss = new ArrayList<Integer>(); 
	
	Map<String, Integer> words = new HashMap<>();
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>(); 
	Map<String, Integer> twords = new HashMap<>(); // �ʵ�
	ArrayList<ArrayList<Double>> tC = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> pinyins = new HashMap<>(); //pinyin map
	ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();
	
	int dim = 100;
	
	String trainfile;
	String evaluatefile;
	
	String filesplit;
	Boolean ispinyin = false;
	String splitchar = "\t";
	double wa, wb;
	
	public Spearman(String file1, String file2, Boolean ispinyin, String splitchar,double wa, double wb) {
		this.trainfile = file1;
		this.evaluatefile = file2;
		this.ispinyin = ispinyin;
		this.splitchar = splitchar;
		this.wa = wa;
		this.wb = wb;
	}
	
	public double computeSpearman() {
		
		readVec();
		readSimword(evaluatefile);
		trainWordsSimiliar();
		
		BigDecimal bd;
		ArrayList<WordPair> trainword = new ArrayList<WordPair>();
		ArrayList<WordPair> fileword = new ArrayList<WordPair>();
		
		for (int i = 0; i < trainwordsim.size(); i++) {
			bd = new BigDecimal(trainwordsim.get(i));
			double t = bd.setScale(3, BigDecimal.ROUND_HALF_UP).doubleValue();
			trainword.add(new WordPair(i, t));
		}
		
		int jk = 0;
		for(int i = 0; i < simword.size(); i++) {
			if(!loss.contains(i)) {
				fileword.add(new WordPair(jk++, Double.valueOf(simword.get(i).get(2))));
			}
		}
		
		Collections.sort(trainword);
		Collections.sort(fileword);
		
		calcuRank(trainword);
		calcuRank(fileword);
		
		double d = 0;
		for(int i = 0; i < fileword.size(); i++) {
			for(int j = 0; j < trainword.size(); j++) {
				if(fileword.get(i).id == trainword.get(j).id) {
					d = d + (fileword.get(i).rank - trainword.get(j).rank) * (fileword.get(i).rank - trainword.get(j).rank);
					break;
				}
			}
		}
		
		double size = fileword.size();
		double spearman = 1 - (6 * d) / (size * (size * size - 1));
		System.out.println("忽略的词" + loss.size());
		System.out.println("参与训练的词" + trainword.size());
		return spearman * 100;
	}
	
	public void calcuRank(ArrayList<WordPair> wordpairs) {
		//����ȼ�	
		double count = 1.0;
		double sum = 1.0;
		
		for (int i = 0; i < wordpairs.size(); i++) {
			if(i + 1 < wordpairs.size() && wordpairs.get(i).sim == wordpairs.get(i + 1).sim) {
				count = count + 1;
				sum = sum + i + 2;
			} else {
				double t = sum / count;
				while(count > 0) {
					wordpairs.get((int) (i - count + 1)).rank = t;
					count--;
				}
				count = 1.0;
				sum = i + 2;
			}
		}
		
	}
	
	void trainWordsSimiliar() {
		double s1;
		SpanishPronunciation esTransform = new SpanishPronunciation();
		int numa = 0, numb = 0, indexp; // �ʶ��ڴʵ��к������
		double theta1a, theta1b, theta2a, theta2b;
		String worda, wordb, espronun;
		Double[] X1a = new Double[dim]; // the first word vec
		Double[] X1b = new Double[dim]; // the second word vec
		Double[] X2a = new Double[dim]; // the first word vec
		Double[] X2b = new Double[dim]; // the second word vec

		for (int pairsize = 0; pairsize < simword.size(); pairsize++) {

			s1 = 0.0;
			theta1a = 0.0;
			theta1b = 0.0;
			theta2a = 0.0;
			theta2b = 0.0;
			worda = simword.get(pairsize).get(0);
			wordb = simword.get(pairsize).get(1);

			for (int i = 0; i < dim; i++) {
				X1a[i] = 0.0;
				X1b[i] = 0.0;
				X2a[i] = 0.0;
				X2b[i] = 0.0;
			}

			if (words.get(worda) == null || words.get(wordb) == null) {
				loss.add(pairsize);
				continue;
			} else {
				numa = words.get(worda);
				numb = words.get(wordb);
				for (int i = 0; i < dim; i++) {
					X1a[i] = X1a[i] + C.get(numa).get(i);
					X1b[i] = X1b[i] + C.get(numb).get(i);
				}
			}

			if(ispinyin) {
				espronun = esTransform.convert_spanish_word_to_phonetic_transcription(worda, "es_ES");
				indexp = pinyins.get(espronun);
				for (int i = 0; i < dim; i++) {
					X1a[i] = X1a[i] + P.get(indexp).get(i);
				}
				
				espronun = esTransform.convert_spanish_word_to_phonetic_transcription(wordb, "es_ES");
				indexp = pinyins.get(espronun);
				for (int i = 0; i < dim; i++) {
					X1b[i] = X1b[i] + P.get(indexp).get(i);
				}
			}

			for (int j = 0; j < dim; j++) {
				theta1a = theta1a + X1a[j] * X1a[j];
				s1 = s1 + X1a[j] * X1b[j];
				theta1b = theta1b + X1b[j] * X1b[j];
			}
			s1 = s1 / (Math.sqrt(theta1b) * Math.sqrt(theta1a));
			trainwordsim.add(s1);
		}
	}
	
	public void readVec() {
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0, pnum = 0, tnum = 0;
			line = read.readLine();
			factors = line.split(" ");
			// wa = Double.parseDouble(factors[0]);
			// wb = Double.parseDouble(factors[1]);
			while ((line = read.readLine()) != null) {
				if (line.equals("the word vector is ")) {
					continue;
				}
				if (line.equals("the pinyin vector is ") || line.equals("the temp vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.put(factors[0], num);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);
				num++;
			}

			while ((line = read.readLine()) != null) {
				if (line.equals("the pinyin vector is ")) {
					break;
				}
				factors = line.split(" ");
				twords.put(factors[0], tnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				tC.add(vec);
				tnum++;
			}

			while ((line = read.readLine()) != null) {
				factors = line.split(" ");
				pinyins.put(factors[0], pnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				P.add(vec);
				pnum++;
			}

			System.out.println("C size " + C.size() + " " + num);
			System.out.println("tC size " + tC.size() + " " + tnum);
			System.out.println("P size " + P.size() + " " + pnum);
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void readSimword(String file) {
		try {
			InputStreamReader in = new InputStreamReader(new FileInputStream(new File(file)), "utf-8");
			BufferedReader read = new BufferedReader(in);
			String line;
			//line = read.readLine();
			String[] split;
			while ((line = read.readLine()) != null) {

				ArrayList<String> temparray = new ArrayList<String>();
				split = line.split(splitchar);
				for (int i = 0; i < split.length; i++) {
					temparray.add(split[i]);
				}
				simword.add(temparray);

			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
