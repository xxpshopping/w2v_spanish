package analogy;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

import test.WeightedWords;

public class Analogy {
	
	Map<String, Integer> words = new HashMap<>();
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>();
	int dim = 100;
	String trainfile;
	int range;
	
	public Analogy(String trainfile, int range) {
		this.trainfile = trainfile;
		this.range = range;
	}
	
	public double accuracyAnalogy() {
		readVec();
		InputStreamReader input;
		double total = 0;
		double right = 0;
		double[] r = new double[dim];
		ArrayList<WeightedWords<String>> top5;
		try {
			input = new InputStreamReader(new FileInputStream("family"), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			while((line = read.readLine()) != null) {
				if(line.equals(": capital-common-countries") || line.equals(": family") || line.equals(": city-in-state")) {
					continue;
				}
				String[] split = line.split(" ");
				Integer w1 = words.get(split[0]);
				Integer w2 = words.get(split[1]);
				Integer w3 = words.get(split[2]);
				Integer w4 = words.get(split[3]);
				if(w1 != null && w2 != null && w3 !=null && w4 != null) {
					for (int i = 0; i < dim; i++) {
						r[i] = C.get(w2).get(i) - C.get(w1).get(i) + C.get(w3).get(i);
					}
					top5 = findSimiliar(r);
					total++;
					for(int i = 0; i < top5.size(); i++) {
						System.out.println(top5.get(i).num);
						if(top5.get(i).num.equals(split[3])) {
							right++;
							break;
						}
					}
					System.out.println("-----------------------");
//					System.out.println(split[0] + " " + split[1] + " " + split[2] + " " + top5.get(0).num + " " + top5.get(1).num + " " + top5.get(2).num + " " + top5.get(3).num + " " + top5.get(4).num);
					
				}
				
			}
		} catch (UnsupportedEncodingException | FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("right:" + right + "total:" + total);
		return right / total * 100;
	}
	
	//get the analogy result
	ArrayList<WeightedWords<String>> findSimiliar(double[] word) {
		double s;
		ArrayList<WeightedWords<String>> top5 = new ArrayList<WeightedWords<String>>();
		double thetaw = 0.0, thetas;
		
		for (int j = 0; j < dim; j++) {
			thetaw = thetaw + word[j] * word[j];
		}
		
		for (Entry<String, Integer> e: words.entrySet()) {
			s = 0.0;
			thetas = 0.0;
			for (int j2 = 0; j2 < dim; j2++) {
				s = s + word[j2] * C.get(e.getValue()).get(j2);
				thetas = thetas + C.get(e.getValue()).get(j2) * C.get(e.getValue()).get(j2);
			}
			s = s / (Math.sqrt(thetas) * Math.sqrt(thetaw));
			top5.add(new WeightedWords<String>(e.getKey(), s));
			Collections.sort(top5);
			if(top5.size() > range) {
				top5.remove(range);
			}
		}
		return top5;
	}
	
	public void readVec() {
		// ��ȡ��ѵ���õĴ�����
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0, pnum = 0, tnum = 0;
			line = read.readLine();
			factors = line.split(" ");

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
			
			System.out.println("C size " + C.size() + " " + num);
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
