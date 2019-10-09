

import java.io.*;
import java.util.*;

@SuppressWarnings("unchecked")
public class Apriori{
	
	public List<int[]> sets;//sets to store all the frequent itemsets
	public int minSupport;//minimum support
	public int min;//minimum value of the input
	public int max;//maximum value of the input 
	public ArrayList<Integer> countSets;//count how many times the item appears
	public ArrayList<Integer>[] storage;//keep track of the lines corresponding to each element of length 1


	public static void main(String[] args) throws Exception{
		Apriori ap=new Apriori(args);
		
	}


	//initialize 
	public Apriori(String[] args) throws Exception{
		long startTime=System.currentTimeMillis();

		PrintStream out = new PrintStream(new FileOutputStream(args[2]));
		System.setOut(out);//output file
		

		sets = new ArrayList<int[]>();
		minSupport=Integer.parseInt(args[1]);
		scale(args);
		storage=new ArrayList[max-min+1];
		for(int i=0; i<max-min+1; i++){
			storage[i]=new ArrayList<Integer>();
		}
		countSets=new ArrayList<Integer>();//initialize all the attributes

		generateL1(args);//generate L1
		for(int i=0; i<sets.size(); i++){
			printArray(sets.get(i));
			System.out.print(" "+"("+countSets.get(i)+")");
			System.out.print("\n");
		}	//output L1
		
		 
		while(!sets.isEmpty()){
			generateNextLevel();
			findFrequent(args);
			for(int i=0; i<sets.size(); i++){
				printArray(sets.get(i));
				System.out.print(" "+"("+countSets.get(i)+")");
				System.out.print("\n");
			}	
		}//use a while loop to generate all levels
		//and output all the itemsets



		long stopTime=System.currentTimeMillis();
		long elapsedTime=stopTime-startTime;//count time
		System.out.println("runtime of the program: ");
		System.out.println(elapsedTime);//output the time
	}


	//search through the database once and find the minimum item and maximum item
	public void scale(String[] args){
		Scanner scan=null;
		try {
			scan = new Scanner(new File(args[0]));
		} catch (FileNotFoundException e) {
			System.out.println("no input file");
		}
		min=Integer.parseInt(scan.next());
		max=Integer.parseInt(scan.next());
		int count = 0;
		while(scan.hasNextLine()){
			count++;
			Scanner scan2=new Scanner(scan.nextLine());
			
			while(scan2.hasNext()){
				String s=scan2.next();
				int i=Integer.parseInt(s);
				if(i<min) min=i;
				if(i>max) max=i;
			}
		}
		System.out.println(count);
	}
	

	//generate all the frequent with length 1 and keep track the lineNum 
	//that contains the frequent item
	public void generateL1(String[] args){
		Scanner scan = null;
		int numLine=1;
		try {
			scan = new Scanner(new File(args[0]));
		} catch (FileNotFoundException e) {
			System.out.println("no input file");
		}
		while(scan.hasNextLine()){
			Scanner scan2=new Scanner(scan.nextLine());
			while(scan2.hasNext()){
				String s=scan2.next();
				int i=Integer.parseInt(s);
				storage[i-min].add(numLine);
			}
			numLine++;
		}//scan through the database and keep track of the LineNum that contains some
		//certain elements
		for(int i=0; i<max-min+1; i++){
			if(storage[i].size()>=minSupport){
				int[] element={min+i};
				sets.add(element);
				countSets.add(storage[i].size());
			}
		}//for each length 1 set, we can check whether it is frequent by counting 
		//the number of lines that contains the set(which is the size of the list 
		//of LineNum), if the number of lines is greater than or equal to the minSupport,
		//the set is frequent and add it to sets
	}
		

	//generate the next level candidate set from the frequent itemsets of current level
	public void generateNextLevel(){
		ArrayList<int[]> result=new ArrayList<int[]>();
		for(int i=0; i<sets.size(); i++){
			for(int j=i+1; j<sets.size(); j++){
				int[] hand1=sets.get(i);
				int[] hand2=sets.get(j);
				boolean match=true;
				for(int k=0; k<hand1.length-1; k++){
					if(hand1[k]!=hand2[k]) match=false;
				}//get two items from the sets 
				//and check whether the first length-1 part are the same
				if(match==true){
					int[] newHand=new int[hand1.length+1];
					for(int x=0; x<hand1.length; x++){
						newHand[x]=hand1[x];
					}
					newHand[hand1.length]=hand2[hand1.length-1];
					result.add(newHand);
				}//if the first length-1 match, we can add them up to make a new item 
				//of length+1
				//we already know the former length-1 part are the same, so copy into
				//the new item, then add the last elem of the former item and then the 
				//latter one b.c the sets is ordered
			}
		}
		sets=result;
	}

	
	//find the frequent set form the candidate sets
	public void findFrequent(String[] args){
		countSets.clear();
		List<int[]> frequent = new ArrayList<int[]>();
		for(int i=0; i<sets.size(); i++){
			int[] current=sets.get(i);
			int c=checkFrequency(current);
			if(c>=minSupport) {
				frequent.add(current);
				countSets.add(c);
			}	
		}//for all the items in the candidates set, check its frequency, 
		//if frequency>=minimum support, the item is frequent, add to the sets
		sets=frequent;
	}


	//check the how many times an item appears by counting the number of the 
	//Lines that contains all of the integer element of the item
	public int checkFrequency(int[] count){
		int countNum=0;
		for(int i=0; i<storage[count[0]-min].size(); i++){//search for all the line
			//of the first integer of the item
			int index=storage[count[0]-min].get(i);
			boolean flag=true;
			for(int j=1; j<count.length; j++){
				if(!storage[count[j]-min].contains(index)) {
					flag=false;
				}
			}//search for other integers of the item whether it has the line
			if(flag==true) countNum++;//if all the integer has the line
			//the line contains all part of the integer, count++
		}
		return countNum;
	}


	public static void printArray(int[] a){
		System.out.print(a[0]);
		for(int i=1; i<a.length; i++){
			System.out.print(" "+a[i]);
		}
		//System.out.print("\n");
	}

}