package test;

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
import java.util.Collections;

import analogy.Analogy;

public class MainTest {

	static ArrayList<String> words = new ArrayList<String>();
	static ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>();
	static int dim = 100;
	static ArrayList<String> pinyins = new ArrayList<String>();
	ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();
	
	//String[] textPath = {"English/EnglishTREC"};
	//String[] textPath = {"English/20_newsgroups"};
	String[] textPath = {"TASS"};
	String[] encode = {"utf-8"};
	//String[] split = {"\t", "\t", " ", "\t"};
	String[] split = {"\t", "\t"};
	
	public static void main(String[] args) {
		boolean isbach = false;
		boolean iswordsim = true;
		boolean isclassify = false;
		boolean isanalogy = false;
		boolean isfind = false;
		boolean ispinyin = false;
		String trainFile = null;
		String resultFile = null;
		String word = "北京";
		int range = 5;
		String model = null;
		double wa = 1, wb = 1;
		
		int j;
		if((j = getParam("-iswordsim", args)) > 0) {
			iswordsim = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-isclassify", args)) > 0) {
			isclassify = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-isanalogy", args)) > 0) {
			isanalogy = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-isfind", args)) > 0) {
			isfind = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-isbach", args)) > 0) {
			isbach = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-ispinyin", args)) > 0) {
			ispinyin = Boolean.valueOf(args[j]);
		}
		if((j = getParam("-input", args)) > 0) {
			trainFile = args[j];
		}
		if((j = getParam("-output", args)) > 0) {
			resultFile = args[j];
		}
		if((j = getParam("-range", args)) > 0) {
			range = Integer.valueOf(args[j]);
		}
		if((j = getParam("-word", args)) > 0) {
			word = args[j];
		}
		if((j = getParam("-model", args)) > 0) {
			model = args[j];
		}
		if((j = getParam("-wa", args)) > 0) {
			wa = Double.parseDouble(args[j]);
		}
		if((j = getParam("-wb", args)) > 0) {
			wb = Double.parseDouble(args[j]);
		}
//		if((j = getParam("-encode", args)) > 0) {
//			this.encode = args[j];
//		}
		System.out.println("isbach " + isbach);
		System.out.println("ispinyin " + ispinyin);
		System.out.println("isclassify: " + isclassify);
		System.out.println("iswordsim: " + iswordsim);
		System.out.println("isanalogy: " + isanalogy);
		System.out.println("input: " + trainFile);
		System.out.println("output: " + resultFile);
		System.out.println("model: " + model);
		System.out.println("wa: " + wa);
		System.out.println("wb: " + wb);
//		System.out.println("textPath: " + textPath);
//		System.out.println("encode: " + encode);
		
		MainTest st = new MainTest();
		if(iswordsim) {
			st.wordSim(isbach, trainFile, resultFile,ispinyin, wa, wb);
		}
		
		if(isclassify) {
			st.textClassification(isbach, trainFile, resultFile);
		}
		
		if(isanalogy) {
			st.analogyEva(isbach, trainFile, resultFile, range);
		}
		
		if(isfind) {
			st.readVec(trainFile);
			st.findSimiliar(word, words);
		}
	}
	
	public void textClassification(boolean isbach, String trainFile, String resultFile) {
		if (isbach) {
			try {
				for (int i = 0; i < textPath.length; i++) {
					File saveResult = new File(resultFile);
					BufferedWriter write = new BufferedWriter(
							new OutputStreamWriter(new FileOutputStream(saveResult), "utf-8"));
					write.write(
							"Evaluation accuracies (%) on text classification, type, precision, recall, f value and total precision\r\n");
					write.flush();

					File dir = new File(trainFile);
					if (dir.isDirectory()) {
						System.out.println("the parameter file is a directory");
					}
					File[] trainfiles = dir.listFiles();
					double s;

					for (File file : trainfiles) {
						if (file.toString().matches(".*txt")) {
							write.write(file.getName() + " ");
							write.flush();

							TextClassfication tc = new TextClassfication(file.toString(), false);
							s = tc.calculateAcc(textPath[i], encode[i], write);
							write.write(String.format("%.4f", s) + "\r\n");
							write.write("\r\n");
							write.flush();
						}
					}
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
//			TextClassfication tc = new TextClassfication("");
//			 double s = tc.calculateAcc(textPath, encode);
//			 System.out.println(s);
		}
	}
	
	public void analogyEva(boolean isbach, String trainFile, String resultFile, int range) {
		if (isbach) {
			try {
				File saveResult = new File(resultFile);
				BufferedWriter write = new BufferedWriter(
						new OutputStreamWriter(new FileOutputStream(saveResult), "utf-8"));
				write.write("Evaluation accuracies (%) on analogical reasoning\r\n");
				write.flush();

				File dir = new File(trainFile);
				if (dir.isDirectory()) {
					System.out.println("the parameter file is a directory");
				}
				File[] trainfiles = dir.listFiles();
				double s;

				for (File file : trainfiles) {
						Analogy analogy = new Analogy(file.toString(), range);
						s = analogy.accuracyAnalogy();
						write.write(file.getName() + " " + s + "\r\n");
						write.flush();
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			String trainfile = "CBOW/1cbow.txt";
			Analogy analogy = new Analogy(trainfile, range);
			System.out.println(analogy.accuracyAnalogy());
		}
	}
	
	public void wordSim(boolean isbach, String trainFile, String resultFile,Boolean ispinyin, double wa, double wb) {
		if (isbach) {
			try {
				File saveResult = new File(resultFile);
				BufferedWriter write = new BufferedWriter(
						new OutputStreamWriter(new FileOutputStream(saveResult), "utf-8"));
				write.write("ES-WS-353\tRG-65\r\n");
				write.flush();

				File dir = new File(trainFile);
				if (dir.isDirectory()) {
					System.out.println("the parameter file is a directory");
				}
				File[] trainfiles = dir.listFiles();
				//String[] testDate = { "en/EN-WS-353-ALL.txt" , "en/EN-RG-65.txt", "en/EN-MEN-TR-3k.txt", "en/EN-RW-STANFORD.txt"};
				String[] testDate = { "es/ES-WS-353.txt" , "es/ES-RG-65.txt"};
				double s;

				for (File file : trainfiles) {
					if (file.toString().matches(".*txt")) {
						for (int i = 0; i < testDate.length; i++) {
							String evaluatefile = testDate[i];
							Spearman spe = new Spearman(file.toString(), evaluatefile, ispinyin, split[i], wa, wb);
							s = spe.computeSpearman();
							if (i == 0) {
								write.write(file.getName() + " " + s);
							} else {
								write.write(" " + s);
							}
							if (i == testDate.length - 1) {
								write.write("\r\n");
							}
							write.flush();
						}
					}
				}
			} catch (UnsupportedEncodingException | FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else {
			String[] testData = { "es/ES-SimLex-999.txt", };
			for (int i = 0; i < testData.length; i++) {
				String evaluatefile = testData[i];
				String trainfile = "1skipgram.txt";
				Spearman spe = new Spearman(trainfile, evaluatefile,ispinyin, "\t", wa, wb);
				System.out.println(testData[i] + " " + spe.computeSpearman());
			}
		}
	}
	
	public static int getParam(String para, String[] args) {
		int i;
		for(i = 0; i < args.length; i++) {
			if(args[i].equals(para)) {
				return i + 1;
			}
		}
		return -1;
	}
	
	void findSimiliar(String word, ArrayList<String> dic) {
		double s;
		ArrayList<WeightedWords<Integer>> top5 = new ArrayList<WeightedWords<Integer>>();
		int i;
		double thetaw = 0.0, thetas;
		
		for (i = 0; i < dic.size(); i++) {
			if(word.equals(dic.get(i))) {
				break;
			}
		}
		
		for (int j = 0; j < dim; j++) {
			thetaw = thetaw + C.get(i).get(j) * C.get(i).get(j);
		}
		for (int j = 0; j < dic.size(); j++) {
			s = 0.0;
			thetas = 0.0;
			for (int j2 = 0; j2 < dim; j2++) {
				s = s + C.get(i).get(j2) * C.get(j).get(j2);
				thetas = thetas + C.get(j).get(j2) * C.get(j).get(j2);
			}
			s = s / (Math.sqrt(thetas) * Math.sqrt(thetaw));
			top5.add(new WeightedWords<Integer>(j, s));
			Collections.sort(top5);
			if(top5.size() > 5) {
				top5.remove(5);
			}
		}
		//��������Ƶ�5����
		for (int j = 0; j < top5.size(); j++) {
			System.out.println(dic.get(top5.get(j).num) + " " + top5.get(j).weight);
		}
	}
	
	public void readVec(String file) {
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(file), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			while((line = read.readLine()) != null) {
				if(line.equals("the word vector is ")) {
					continue;
				}
				if(line.equals("the pinyin vector is ") || line.equals("the Character vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.add(factors[0]);
				ArrayList<Double> vec = new ArrayList<Double>();
				for(int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);
			}
			while((line = read.readLine()) != null) {
				factors = line.split(" ");
				pinyins.add(factors[0]);
				ArrayList<Double> vec = new ArrayList<Double>();
				for(int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				P.add(vec);
		}
			System.out.println("C size " + C.size());
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
}
