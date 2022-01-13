from selenium import webdriver
import time
import pandas

year=2009
month=1
day=1

iteration=1000000

EQUIPE1=[]
EQUIPE2=[]
DATE=[]
SCORE1=[]
SCORE2=[]



def AddDay():
    global day
    global month
    global year
    TrenteOuUn=0

    if month in [1,3,5,7,8,10,12]:
        TrenteOuUn=31
    elif month==2:
        TrenteOuUn=28
    else:
        TrenteOuUn=30



    if day>=TrenteOuUn:
        month+=1
        day=1
    else:
        day+=1

    if month>12:
        year+=1
        month=1


    print(year, month, day)


def Addthezero(number):
    if number<10:
        return str("0"+str(number))
    return str(number)

def main():
    global iteration
    while iteration>0 :
        print(1000000-iteration)
        iteration-=1

        AddDay()
        browser.get("https://www.nba.com/games?date="+ str(year) + "-"+ Addthezero(month) + "-"+ Addthezero(day)) #"https://www.nba.com/games?date=2009-04-15"

        #print("__DATE__",day, month, year)
        input_equipe=browser.find_elements_by_css_selector(".MatchupCardTeamName_teamName__3i23P")
        for i in range(len(input_equipe)):
            #print("Equipe : ",input[i].text)
            if i%2==0:
                EQUIPE1.append(input_equipe[i].text)
                DATE.append( str(year) + "-"+ Addthezero(month) + "-"+ Addthezero(day))
            else:
                EQUIPE2.append(input_equipe[i].text)

        input=browser.find_elements_by_css_selector(".h9.relative.inline-block.leading-none")
        for i in range(len(input_equipe)):
            #print("SCORE1 : ",input[i].text)
            if i%2==0:
                try :
                    SCORE1.append(input[i].text)
                except:
                    SCORE1.append("None")
            else:
                try:
                    SCORE2.append(input[i].text)
                except:
                    SCORE2.append("None")

browser=webdriver.Chrome()


def boucle():
    try:
        main()

    except:
        browser=webdriver.Chrome()
        boucle()

boucle()



ar = []
ar.append(DATE)
ar.append(EQUIPE1)
ar.append(EQUIPE2)
ar.append(SCORE1)
ar.append(SCORE2)

df = pandas.DataFrame(ar , index = ['DATE','EQUIPE1', 'EQUIPE2', 'SCORE1', 'SCORE2']) #
df = df.transpose()
print(df)


df.to_csv("datasets/ScoreEquipeNBA.csv")
df.to_excel("datasets/ScoreEquipeNBA.xlsx")
