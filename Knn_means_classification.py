import pandas as pnd
import time
import csv
import re
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
from collections import Counter
import shelve
import math
import pdb
import enchant
import numpy
#import dbm.dumb as dumbdbm
from collections import OrderedDict
#db_s=dumbdbm.open("shlvobject.txt")
shelve_tfdf=shelve.open("shlvobject.txt")
#shelve_tfdf=shelve.Shelf(db_s)
#db_c=dumbdbm.open("cossim.txt")
cossim_train=shelve.open("cossim.txt")
#cossim_train=shelve.Shelf(db_c)

train_csv = pnd.read_csv('train.csv', encoding="ISO-8859-1")
stemmer=PorterStemmer()

prodDesc_csv = pnd.read_csv('product_descriptions.csv',encoding="ISO-8859-1")
test_csv = pnd.read_csv('test.csv', encoding="ISO-8859-1")
attr_csv =pnd.read_csv('attributes.csv', encoding="ISO-8859-1")
attr_csv_brand =attr_csv[attr_csv.name=="MFG Brand Name"][["product_uid","value"]]
train_csv=pnd.merge(train_csv,prodDesc_csv,left_index=True,on='product_uid')
train_csv=pnd.merge(train_csv,attr_csv_brand,on='product_uid',how='left')
test_csv=pnd.merge(test_csv,prodDesc_csv,left_index=True,on='product_uid')
test_csv=pnd.merge(test_csv,attr_csv_brand, on='product_uid',how='left')
pwl = enchant.request_pwl_dict("hd_spellcheck.txt")
word_dict=enchant.request_dict("en_US")
produid_tf={}
produid_tf=shelve_tfdf['produid_tf']

length_of_vector={}
length_of_vector=shelve_tfdf['length_of_vector']


cossim={}
cossim=cossim_train["cossim"]
def preprocess(sline,addDict):
	sline = sline.replace("-"," ")
	sline = sline.replace("&amp;"," and ");
	procLine=re.sub("(?<=\d),(?=\d)","",sline)
	procLine=re.sub("(?<=\d)[\s]+(inches|inch|in\.*)","inch ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(gallons|gallon|gal\.*)","gallon ",procLine)
	procLine=re.sub("(?<=\d)[\s]sq\.[\s]ft\." , "sqft ",procLine)
	procLine=re.sub("(?<=\d)[\s]cu\.[\s]ft\." , "cuft ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(feet|foot|ft\.*)","feet ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(amperes|ampere|amp\.*|amps)","ampere ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(lbs\.*|lb\.*|pound|pounds)","pound ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(ounces|ounce|oz\.*)","oz ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(cc)","cc ",procLine)
	procLine=procLine.replace("''","inch ")
	procLine=procLine.replace("'","feet ")
	procLine=procLine.replace("°","degree ")
	procLine=re.sub("[\s]v[\s]"," volt ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(degrees|degree|deg\.*)","degree ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(volts|volt)","volt ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(watts|watt)","watt ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(mi[l]{1,2}imeter[s]*|mm|mm\.*)","mm ",procLine)
	procLine=re.sub("(?<=\d)[\s]+(centimeter[s]*|cm|cm\.*)","cm ",procLine)
	procLine=re.sub(r"([\d]+\.*[\d]*)([x|X])([\d]+\.*[\d]*)([x|X])([\d]+\.*[\d]*)([a-z]+)",r"\1\6  x \3\6 x \5\6 ",procLine)
	#print(procLine+"\n")
	procLine=procLine.replace("RPM","RPM ");
	procLine=procLine.replace("rpm","rpm ");
	#https://docs.python.org/2/library/re.html?highlight=regular%20expression
	#Removing all periods except the decimal points
	procLine=re.sub("(?<=\D){1,}[.](?=\D)*[\w]*"," ",procLine)
	#Removing commas in between numbers
#	procLine=re.sub("(?<=\d),(?=\d)","",procLine)
	#Removing all/ except on numbers
	procLine=re.sub(r'\/(?=\D)',' ',procLine)
	#Removing all special character except some
	procLine=re.sub(r'[^a-zA-Z0-9.\/]',' ',procLine)
	#print(procLine+"\n")
	wTokens=word_tokenize(procLine)
	stopw=stopwords.words('english')
	#Add product features token to personal dictionary for spell check of search terms
	if(addDict):
		for token in wTokens:
			if token not in stopw and len(token)>1 and token.isdigit()==False and pwl.is_added(token)==False:
				pwl.add(token)
	p_wTokens=[stemmer.stem(token) for token in wTokens if token not in stopw]
	return p_wTokens

def calcTfTest(csv_doc):
	for uid in csv_doc["product_uid"]:
		if(produid_tf.get(uid)==None):
			ptitle=csv_doc[csv_doc.product_uid== uid]["product_title"].values[0]
			p_wtokens=preprocess(ptitle.lower(),True)
			produid_tf[uid]={}
			produid_tf[uid]['product_title']={}
			produid_tf[uid]['product_title']=nltk.FreqDist(p_wtokens)
			
			prodesc=csv_doc[csv_doc.product_uid== uid]["product_description"].values[0]
			
			prodesc=re.sub(r'(?=[A-Z][a-z])',' ',prodesc)
			#Add whitespace between character and number	
			prodesc=re.sub(r"(?<=[a-z])([\d]+)",r" \1",prodesc)
			prodesc=re.sub(r"(?<=[\d])([a-z]+)",r" \1",prodesc)
			prodesc=re.sub(r"([.])(?=[A-Z])",r"\1 ",prodesc)
			#remove all periods at end of sentences
			prodesc=re.sub(r"(?<=[\D][\s])([a-zA-Z]{3,})[.]",r"\1 ",prodesc)
			#Add whitespace between two words
			prodesc=re.sub(r"(?<=[a-z])([A-Z])(?=[a-z])",r" \1",prodesc)
			
			pd_wtokens=preprocess(prodesc.lower(),True)
			
			produid_tf[uid]['product_description']={}
					
			produid_tf[uid]['product_description']=nltk.FreqDist(pd_wtokens)
		
			prodbrand=csv_doc[csv_doc.product_uid==uid]["value"].values[0]
			pbrand_wtokens=preprocess(str(prodbrand).lower(),True)
			produid_tf[uid]['brand_name']={}
			
			produid_tf[uid]['brand_name']=nltk.FreqDist(pbrand_wtokens)
			
			#calculate logarithmic tf
			length_of_vector[uid]={}
			pt_tf=produid_tf[uid]['product_title']
			sumv=0
			for term in pt_tf:
				produid_tf[uid]['product_title'][term]=1+math.log10(pt_tf[term])
				sumv+=produid_tf[uid]['product_title'][term]**2
			length_of_vector[uid]['product_title']=math.sqrt(sumv)

			sumv=0
			pd_tf=produid_tf[uid]['product_description']

			for term in pd_tf:
				produid_tf[uid]['product_description'][term]=1+math.log10(pd_tf[term])
				sumv+=(produid_tf[uid]['product_description'][term])**2
			length_of_vector[uid]['product_description']=math.sqrt(sumv)


			sumv=0
			pb_tf=produid_tf[uid]['brand_name']
			for term in pb_tf:
				produid_tf[uid]['brand_name'][term]=1+math.log10(pb_tf[term])
				sumv+= (produid_tf[uid]['brand_name'][term])**2
			length_of_vector[uid]['brand_name']=math.sqrt(sumv)
			
		print(uid)


def calculate_cossim(search_tf,length_search,uid):
	product_data=produid_tf[uid]
	sum_cossim=0
	csim_pt=0
	csim_pd=0
	csim_pb=0
	#Calculate dot product numerator of  cosine similarity of search term with prod title, description,brand name
	for term in search_tf:
		if product_data["product_title"].get(term):
			csim_pt+=product_data["product_title"].get(term)*search_tf[term]
		if product_data["product_description"].get(term):
			csim_pd+=product_data["product_description"].get(term)*search_tf[term]
		if product_data["brand_name"].get(term):
			csim_pb+=product_data["brand_name"].get(term)*search_tf[term]

	sum_cossim=(csim_pt/(length_search*length_of_vector[uid]["product_title"]))+(csim_pd/(length_search*length_of_vector[uid]["product_description"]))
	if length_of_vector[uid]["brand_name"]!=0:		
		sum_cossim+=(csim_pb/(length_search*length_of_vector[uid]["brand_name"]))
	return sum_cossim

def stCleaning(search_terms):
	search_terms=re.sub("(?<=\d){1,}[\s]*(in|in\.|inch)[\s]"," inch ",search_terms)
	search_terms=re.sub("(?<=\d){1,}[\s]*(ft|ft\.|feet)[\s]"," feet ",search_terms)
	search_terms=re.sub("(?<=\d){1,}[\s]*(amp|amp\.|ampere)[\s]"," ampere ",search_terms)
	search_terms=re.sub("(?<=\d){1,}[\s]*(lb|lbs\.|pound)[\s]"," pound ",search_terms)
	search_terms=re.sub("(?<=\d){1,}[\s]*(gal|gal\.|gallon)[\s]"," gallon ",search_terms)
	search_terms=re.sub(r"([\d]+\.*[\d]*)([x|X])",r"\1inch \1feet \2",search_terms)
	search_terms=re.sub(r"(?<=[x|X])([\s]*[\d]+\.*[\d]*[^a-z]*)",r" \1inch \1feet ",search_terms)
	search_terms=re.sub(r"(?<=[x|X])([\s]*)([\d]+[\/][\d]+)",r"\1\2inch \2feet ",search_terms)
	#Removing all periods except the decimal points
	search_terms=re.sub(r'\.(?!\d)',' ',search_terms)
	#remove all special characters
	search_terms=re.sub(r'[^a-zA-Z0-9.\/]',' ',search_terms)
	return search_terms

#http://stackoverflow.com/questions/23681948/get-index-of-closest-value-with-binary-search
def binarySearch(data, val):
	highIndex = len(data)-1
	lowIndex = 0
	while highIndex > lowIndex:
		index = int((highIndex + lowIndex) / 2)
		sub = data[index]
		if data[lowIndex] == val:
			return [lowIndex, lowIndex]
		elif sub == val:
			return [index, index]
		elif data[highIndex] == val:
                    return [highIndex, highIndex]
		elif sub > val:
			if highIndex == index:
				return sorted([highIndex, lowIndex])
			highIndex = index
		else:
			if lowIndex == index:
				return sorted([highIndex, lowIndex])
			lowIndex = index
	return sorted([highIndex, lowIndex])

def calculateCossim_train():
	for curr_id in train_csv["id"]:
		print(curr_id)
		uid=train_csv[train_csv.id==curr_id]["product_uid"].values[0]
		search_terms =train_csv[train_csv.id==curr_id]["search_term"].values[0].lower()
		relv_val =train_csv[train_csv.id==curr_id]["relevance"].values[0]
		print(search_terms)
		search_terms=stCleaning(search_terms)
		search_words=word_tokenize(search_terms)
		stopw=stopwords.words('english')
		#Spell check of search term
		for token in search_words:
			#if has number or a stopword or present in the personal dictionary dont spell check
			if re.match("\d",token) or token in stopw or pwl.check(token) or len(token)==1 or re.match(r".*\/.*",token):
				continue
			elif word_dict.suggest(token):
				if " " in word_dict.suggest(token)[0]:
					new_token=word_dict.suggest(token)[0]
					search_terms=search_terms.replace(token,new_token)
				else:
					sugtns=pwl.suggest(token)
					if(sugtns):
						w_sugtn=sugtns[0]
						#if relvance score is more than 2 then choose suggestions from prod desc	
						if len(sugtns)>1 and relv_val>=2:
							token_counter=Counter(token)
							for word in sugtns:
								if(stemmer.stem(word) in produid_tf[uid]['product_description']):
									w_sugtn=word
									break
						search_terms=search_terms.replace(token,w_sugtn)
		print(search_terms)	
	
		search_tokens=preprocess(search_terms,False)
		#if search term contaisn only stopwords
		if len(search_tokens)==0:
			search_tokens=word_tokenize(train_csv[train_csv.id==curr_id]["search_term"].values[0].lower())
		search_tf=Counter(search_tokens)
		length_search=0
		for term in search_tf:
			search_tf[term]=1+math.log10(search_tf[term])
			length_search+=(search_tf[term])**2
		length_search=math.sqrt(length_search)
		cossim[curr_id]=calculate_cossim(search_tf,length_search,uid)

def classify_test():
	#http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
	sorted_cossim=OrderedDict(sorted(cossim.items(), key=lambda x: x[1]));
	scossim_keys=list(sorted_cossim.keys())
	scossim_vals=list(sorted_cossim.values())
	#	with open('fast2_knnClassf.csv',encoding="ISO-8859-1",mode='w') as csvFile:
	csvFile= open('biased_knnClassf.csv',encoding="ISO-8859-1",mode='w')
	csvFile.write("id,relevance\n")
	count=0
	final_result=""
	pp_time=time.time()
	for curr_id in test_csv["id"]:
		stopw=stopwords.words('english')
		uid=test_csv[test_csv.id==curr_id]["product_uid"].values[0]
		search_terms =test_csv[test_csv.id==curr_id]["search_term"].values[0].lower()
		#print(search_terms)
		search_terms =stCleaning(search_terms)
		search_words=word_tokenize(search_terms)
		#Spell check of search term
		for token in search_words:
		#if has number or a stopword or present in the personal dictionary dont spell check
			if re.match("\d",token) or token in stopw or pwl.check(token) or len(token)==1 or re.match(r".*\/.*",token):
				continue
			elif word_dict.suggest(token):
				if " " in word_dict.suggest(token)[0]:
					new_token=word_dict.suggest(token)[0]
					search_terms=search_terms.replace(token,new_token)
				else:
					sugtns=pwl.suggest(token)	
					if(sugtns):
						w_sugtn=sugtns[0]
						search_terms=search_terms.replace(token,w_sugtn)
		#print(search_terms)
		search_tokens=preprocess(search_terms,False)
		if len(search_tokens)==0:
			final_result+=str(curr_id)+","+"1"+"\n"
			continue
		search_tf=Counter(search_tokens)
		length_search=0
		for term in search_tf:
			search_tf[term]=1+math.log10(search_tf[term])
			length_search+=(search_tf[term])**2
		length_search=math.sqrt(length_search)		
		curr_cossim_test=calculate_cossim(search_tf,length_search,uid)
		min_vals=[]
		if(curr_cossim_test==0):
			#min_vals.append(scossim_keys[0])
			#count+=1
			#print("Count:"+str(count))
			#print(search_terms)
			#print(curr_id)
			#for ix in range(1,10):
			#	min_vals.append(scossim_keys[ix])
			for key_id in sorted_cossim:
				if sorted_cossim[key_id]==0:
					train_uid=train_csv[train_csv.id==key_id]["product_uid"].values[0]
					if(train_uid==uid):
						min_vals.append(key_id)
						if(len(min_vals)==3):
							break
				else:
					for ix in range(1,10):
						min_vals.append(scossim_keys[ix])
					break
		else:
			continue
#			closest=binarySearch(scossim_vals,curr_cossim_test)
#			closest1 =closest[0]
#			if closest1>=5 and closest1<len(scossim_keys)-5:
#				min_vals.append(scossim_keys[closest1])
#				for ix in range(1,6):
#					min_vals.append(scossim_keys[closest1-ix])
#					min_vals.append(scossim_keys[closest1+ix])
#			elif closest1<=5:
#				for ix in range(0,10):
#					min_vals.append(scossim_keys[ix])
#			elif closest1>=len(scossim_keys)-5:
#				for ix in range(1,11):
#					min_vals.append(scossim_keys[len(scossim_keys)-ix])
			closest_cossim={}
			for train_id in min_vals:
				closest_cossim[train_id]=math.fabs(cossim[train_id]-curr_cossim_test)
			min_vals=(sorted(closest_cossim,key=closest_cossim.get,reverse=False)[:5])
		relevanceVals=[]
		for train_id in min_vals:
			relevanceVals.append(train_csv[train_csv.id== train_id]["relevance"].values[0])
		result=str(curr_id)+","+str(numpy.mean(relevanceVals))+"\n"
#		relv_list=[]
#		for relv in relevanceVals:
#			if(relv<=1.3):
#				relv_list.append(1)
#			elif(relv>1.3 and relv<=1.7):
#				relv_list.append(1.5)
#			elif(relv<=2.3 and relv>1.7):
#				relv_list.append(2)
#			elif(relv>2.3 and relv<=2.75):
#				relv_list.append(2.5)
#			elif(relv>2.75):
#				relv_list.append(3)
		
#		relv_count=Counter(relv_list)
#		vals=list(relv_count.values())
#		keys=list(relv_count.keys())
#		result=str(curr_id)+","+str(keys[vals.index(max(vals))])+"\n"
		final_result+=result
		#csvFile.write(result)
#		print(result)
#		count=count+1
#		if count==100:
#			break
		
	print("Cossim time taken",time.time()-pp_time)
	csvFile.write(final_result)
	csvFile.close()
if __name__ == '__main__':
	t=time.time()
#	print("Started training")
#	calcTfTest(train_csv)
#	print("train_done\n")
#	print("Processing test")
#	calcTfTest(test_csv)
#	print("Time taken:")
#	print(time.time()-t)
#	shelve_tfdf["produid_tf"]=produid_tf
#	shelve_tfdf["length_of_vector"]=length_of_vector
#	shelve_tfdf.close()
#	calculateCossim_train()
#	print("Cossim train done\n")
#	print(time.time()-t)
#	cossim_train["cossim"]=cossim
#	cossim_train.close()
#	print("Starting Classification of test data")
	currtime=time.time()
	classify_test()
	print("Done Classification")
	print(time.time()-currtime)
