import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fcluster
from math import floor

def Unneeded_Row(Activity_Name, 
                 activity_purpose, 
                 means_of_transportation, 
                 visit_purpose_category,
                 country_area):
    
  if (activity_purpose=='Work / Work related / Study / Volunteering' or
      activity_purpose=='Arrival /Departure Via Ben Gurion Airport' or
      activity_purpose=='Located in Jordan / Egypt' or
      activity_purpose=='Health Services' or
      activity_purpose=="pick up/let off passenger"
      ):
     return True 
  if (activity_purpose=='Changing means of transportation' and
      means_of_transportation != 'Other'
     or u'שדה'  in Activity_Name
       or u'נחיתה'  in Activity_Name
       or u'לא ידוע'  in Activity_Name
         or u'החזרת'  in Activity_Name
          or u'ירידה'  in Activity_Name
          or u'רכבת'  in Activity_Name
      ):
     return True 
  if (visit_purpose_category=='business' and
      not (activity_purpose=='Shopping / Arrangements / Food' or 
           activity_purpose=='Tourism / leisure / religious services / sports activities')):
     return True
  if (activity_purpose=='Shopping / Arrangements / Food' and
      (u'אוכל'  in Activity_Name or u'איסוף' in Activity_Name or
       u'משטרה' in Activity_Name or u'חנייה' in Activity_Name or
       u'חניה' in Activity_Name or  u'גבול' in Activity_Name or
       u'מקדונלדס' in Activity_Name or u'מחסום' in Activity_Name or
       u'תדלוק' in Activity_Name or u'מאפיה' in Activity_Name or
       u'טייק' in Activity_Name or u'מאפיה' in Activity_Name or
       u'תדלוק' in Activity_Name or u'איקאה' in Activity_Name or
       u'מזכירות' in Activity_Name or u'נתבג' in Activity_Name or
       u'פיצרייה' in Activity_Name or u'סוכנות' in Activity_Name or
       u'המתנה' in Activity_Name or u'גן ילדים' in Activity_Name or
       u'בית ספר' in Activity_Name or u'חומוס' in Activity_Name or
       u'ירקן' in Activity_Name or u'מרכז קניות' in Activity_Name or
       u'נתב"ג' in Activity_Name or u'אוטובוס' in Activity_Name or
       u'תעשייה' in Activity_Name or u'טרמינל' in Activity_Name or
       u'כספים' in Activity_Name or u'כושר' in Activity_Name or
       u'תעשייה' in Activity_Name or u'טרמינל' in Activity_Name or
       u'ביג' in Activity_Name or u'יוגורט' in Activity_Name or
       u'שופרסל' in Activity_Name or u'שווארמה' in Activity_Name or
       u'משרד' in Activity_Name or u'נדל"ן' in Activity_Name or
       u'תעשייה' in Activity_Name or u'טרמינל' in Activity_Name or
       u'ביקורת' in Activity_Name or u'מודיעין' in Activity_Name or
       u'תעשייה' in Activity_Name or u'טרמינל' in Activity_Name or
       u'מכולת' in Activity_Name or u'קפה' in Activity_Name or
       u'גלידרייה' in Activity_Name or u'גלידה' in Activity_Name or
       u'סופר' in Activity_Name or u'מאפייה' in Activity_Name or
       u'מסעדת' in Activity_Name or u'רכבת' in Activity_Name or
       u'תחנת' in Activity_Name or u'רכב' in Activity_Name or
       u'בנק' in Activity_Name or u'דלק' in Activity_Name or
       u'דואר' in Activity_Name or u'קיוסק' in Activity_Name or
       u'חנות' in Activity_Name or u'סידורים'or u' מסעדה' in Activity_Name or
       u'רענון' in Activity_Name or u'עצירת' in Activity_Name or
       u'עצירה' in Activity_Name or u'הפסקה' in Activity_Name or
       u'כספומט' in Activity_Name or u'מרכז מסחרי' in Activity_Name or
       u'כסף' in Activity_Name or u'פיצה' in Activity_Name or
       u'חומוס' in Activity_Name or u'פלאפל' in Activity_Name or
       u'סופר' in Activity_Name or u'דוכן' in Activity_Name 
       )):
     return True
 
  if (activity_purpose=='Tourism / leisure / religious services / sports activities' and
      (u'אוכל'  in Activity_Name 
       or u'חניה' in Activity_Name 
       or u'משטרה' in Activity_Name 
       or u'חניון' in Activity_Name 
       or u'מגורים' in Activity_Name 
       or u'מסעדה' in Activity_Name 
       or u'כלב'  in Activity_Name   
       or u'מכר'in Activity_Name
       or u'מכרים' in Activity_Name
       or u'נוסע'  in Activity_Name 
       or u'לא ידוע'  in Activity_Name 
       or u'פגישה' in Activity_Name
       or u'מלון' in Activity_Name 
       or u'חולים'  in Activity_Name 
       or u'אח'  in Activity_Name 
       or u'קפיטריה' in Activity_Name 
       or u'קניון'  in Activity_Name 
       or u'איסוף'  in Activity_Name 
       or u'ציבורי' in Activity_Name 
       or u'סופר' in Activity_Name
       or u'חתונה'  in Activity_Name 
       or u'שדה'  in Activity_Name
       or 'change'  in Activity_Name 
       or u'חניה'  in Activity_Name 
       or u'גבול' in Activity_Name
       or u'ריצה' in Activity_Name
       or u'תחילת טיול' in Activity_Name
        or u'אירוע' in Activity_Name
        or u'קיוסק' in Activity_Name 
          or u'חדר כושר' in Activity_Name 
          or u'חנות' in Activity_Name
           or u'מרכז קניות' in Activity_Name
            or u'טקס' in Activity_Name
      or u'חתונה' in Activity_Name
           or u'בית כנסת' in Activity_Name
              or u'ישיבה' in Activity_Name
                or u'צילום' in Activity_Name
                 or u'תפילת' in Activity_Name
                   or u'נסיעה' in Activity_Name
                     or u'כנס' in Activity_Name
                      or u'מסיבת' in Activity_Name
                          or u'תמונות' in Activity_Name
                          or u'סיום סיור' in Activity_Name
                           or u'המשך סיור' in Activity_Name
                            or u'תחילת סיור' in Activity_Name
                               or u'גלידריה' in Activity_Name
                               or u'שירותי דת' in Activity_Name
                 or u'בית קפה' in Activity_Name
                 or u'לימודים' in Activity_Name
                  or u'מונית' in Activity_Name
                  or u'הסעה' in Activity_Name
                   or u'עצירה' in Activity_Name
                   or u'עצירת' in Activity_Name
                   or u'מדרש' in Activity_Name
                    or u'קאנטרי' in Activity_Name
            or u'קאונטרי' in Activity_Name
             or u'מאפייה' in Activity_Name
             or u'סידורים' in Activity_Name
              or u'אוטו' in Activity_Name
               or u'שיורותי דת' in Activity_Name
               or u'אמצע' in Activity_Name
               or u'שירוי דת' in Activity_Name
                  or u'אוניברסיטה' in Activity_Name
                   or u'טכניון' in Activity_Name  
                   or u'דלק' in Activity_Name
                   or 'am' in Activity_Name
                     or u'אולם' in Activity_Name
                      or u'אולמי' in Activity_Name
                         or  u'אולפנה' in Activity_Name
                          or  u'אוניברסיטת' in Activity_Name
                          or  u'כספומט' in Activity_Name
                             or u'המתנה' in Activity_Name
                     or u'המתין' in Activity_Name
                       or u'תיירות' in Activity_Name
                        or u'ישיבת' in Activity_Name           
       )):
     return True
 
  if (activity_purpose=='Hotel / Holiday house / Residence' and
      (u'מגורים'  in Activity_Name or u'חניה' in Activity_Name or
       u'מלון' in Activity_Name or
       u'בית' in Activity_Name
        or u'איסוף' in Activity_Name
        or u'מגוורים' in Activity_Name
         or u'מכר' in Activity_Name
           or u'מסעדה' in Activity_Name
             or u'בסיס' in Activity_Name
               or u'קיוסק' in Activity_Name
               or u'דלק' in Activity_Name
                or u'שומר' in Activity_Name
                  or u'אכסניה' in Activity_Name
                    or u'אכסנייה' in Activity_Name
                     or u'חנות' in Activity_Name
                     or u'המתנה' in Activity_Name
                     or u'המתין' in Activity_Name
                     or u'ביקור' in Activity_Name
        
        )):
     return True
  
  if (activity_purpose=='Family visit / friends' and
      (u'מגורים'  in Activity_Name 
       or u'בית' in Activity_Name
       or u'ביקור' in Activity_Name
        or u'הורדה' in Activity_Name
        or u'מסעדה' in Activity_Name
        or u'חניה' in Activity_Name
        or u'דירה של' in Activity_Name
        or u'מגוורים' in Activity_Name
        or u'בית ה' in Activity_Name
        or u'אירוע' in Activity_Name
        or u'ליווי מכר' in Activity_Name
        or u'מלון' in Activity_Name
        or u'עבודה' in Activity_Name
        or u'מסיבה' in Activity_Name
        or u'מסיבת' in Activity_Name
        or u'מכולת' in Activity_Name
         or u'מדרשה' in Activity_Name
         or u'ישיבה' in Activity_Name
         or u'קאנטרי' in Activity_Name
          or u'חתונה' in Activity_Name
          or u'פגישה' in Activity_Name
        or u'איסוף' in Activity_Name
         or u'מכר' in Activity_Name         
         or u'ליווי' in Activity_Name
          or u'אולם' in Activity_Name
            or u'אולמי' in Activity_Name
             or u'אזכרה' in Activity_Name
             or u'איכילוב' in Activity_Name
              or u'אייקאה' in Activity_Name
               or u'ארוחת' in Activity_Name
               or u'משפחתי' in Activity_Name
                or u'ארוע' in Activity_Name
         or u'אירוע' in Activity_Name
        
       )):
     return True
     

  if country_area=="Not in Israel":
        return True
  return False
    
def Update_from_Tourist_Area_layer_Source(Tourist_Area_layer_Source):
    if (Tourist_Area_layer_Source=='-1'):
        return False

    if (Tourist_Area_layer_Source.startswith(u'רמת הנדיב') or
       Tourist_Area_layer_Source.startswith(u'ויה דלרוזה') or
        Tourist_Area_layer_Source.startswith(u'כנסיית סנט פטרוס') or
        Tourist_Area_layer_Source.startswith(u'כנסיית המולד וסביבתה') or
        Tourist_Area_layer_Source.startswith(u'כנסיית הקבר הקדוש וסביבתה') or
        Tourist_Area_layer_Source.startswith(u'קבר הגן (גן הקבר)') or
        Tourist_Area_layer_Source.startswith(u'פארק רמות מנשה') or
        Tourist_Area_layer_Source.startswith(u'פארק קצרין העתיקה') or
        Tourist_Area_layer_Source.startswith(u'פארק איילון קנדה') or
        Tourist_Area_layer_Source.startswith(u'ספארי') or
        Tourist_Area_layer_Source.startswith(u'נמל') or
        Tourist_Area_layer_Source.startswith(u'משכנות שאננים וימין משה') or
        Tourist_Area_layer_Source.startswith(u'משכן הכנסת') or
        Tourist_Area_layer_Source.startswith(u'מערת המכפלה') or
        Tourist_Area_layer_Source.startswith(u'מיני ישראל') or
        Tourist_Area_layer_Source.startswith(u'יד לשריון') or
        Tourist_Area_layer_Source.startswith(u'ויה דלרוזה') or
        Tourist_Area_layer_Source.startswith(u'הרודיון') or
        Tourist_Area_layer_Source.startswith(u'המצפה התת ימי') or
        Tourist_Area_layer_Source.startswith(u'הכותל המערבי') or
        Tourist_Area_layer_Source.startswith(u'גן סאקר') or
        Tourist_Area_layer_Source.startswith(u'גן החיות') or
        Tourist_Area_layer_Source.startswith(u'גבעת התחמושת') or
        Tourist_Area_layer_Source.startswith(u'בית התפוצות') or
        Tourist_Area_layer_Source.startswith(u'בית הקברות הצבאי הר הרצל') or
        Tourist_Area_layer_Source.startswith(u'בית אהרונסון') or
        Tourist_Area_layer_Source.startswith(u'אתר החרמון') or
        Tourist_Area_layer_Source.startswith(u'אחוזת קבר דוד בן גוריון') or
        Tourist_Area_layer_Source.startswith(u'עיר דוד') or
        Tourist_Area_layer_Source.startswith(u'יד ושם') or
        Tourist_Area_layer_Source.startswith(u'תיאטרון') or
        Tourist_Area_layer_Source.startswith(u'שוק') or
        Tourist_Area_layer_Source.startswith(u'מוזיאון') or
        Tourist_Area_layer_Source.startswith(u'כיכר') or
        Tourist_Area_layer_Source.startswith(u'חמי') or
        Tourist_Area_layer_Source.startswith(u'יער') or
        Tourist_Area_layer_Source.startswith(u'העיר העתיקה') or
        Tourist_Area_layer_Source.startswith(u'איצטדיון') or
          Tourist_Area_layer_Source.startswith(u'אצטדיון') or
        Tourist_Area_layer_Source.startswith(u'מתחם') or
        Tourist_Area_layer_Source.startswith(u'שמורת טבע') or
        Tourist_Area_layer_Source.startswith(u'גן לאומי') or
        Tourist_Area_layer_Source.startswith(u'הר הקפיצה') or
        Tourist_Area_layer_Source.startswith(u'הר הצופים') or
        Tourist_Area_layer_Source.startswith(u'קסר אליהוד') or
        Tourist_Area_layer_Source.startswith(u'קומראן') or
        Tourist_Area_layer_Source.startswith(u'יריחו') or
        Tourist_Area_layer_Source.startswith(u'הר הבית') or
        Tourist_Area_layer_Source.startswith(u'אל-עזריה (קבר לזרוס)') or
        Tourist_Area_layer_Source.startswith(u'קבר הגן (גן הקבר)') or
        Tourist_Area_layer_Source.startswith(u'רמאללה') or
        Tourist_Area_layer_Source.startswith(u'בית לחם') or
        Tourist_Area_layer_Source.startswith(u'בית לחם') or
        Tourist_Area_layer_Source.startswith(u'משכנות שאננים') or
        Tourist_Area_layer_Source.startswith(u'מרכז העיר') or
         Tourist_Area_layer_Source.startswith(u'נווה צדק') or
         Tourist_Area_layer_Source.startswith(u'הר ציון') or
         Tourist_Area_layer_Source.startswith(u'טיילת ארמון הנציב') or
         Tourist_Area_layer_Source.startswith(u'מגדל דוד') or
         Tourist_Area_layer_Source.startswith(u'יד לשריון') or
         Tourist_Area_layer_Source.startswith(u'צריף בן גוריון') or
        Tourist_Area_layer_Source.startswith(u'מוזיאון') or
        Tourist_Area_layer_Source.startswith(u'פארק הירקון') or
        Tourist_Area_layer_Source.startswith(u'המשכן לאומנויות הבמה') or
        Tourist_Area_layer_Source.startswith(u'נמל') or
        Tourist_Area_layer_Source.startswith(u'עין שבע') or
        Tourist_Area_layer_Source.startswith(u'דולפנים') or
        Tourist_Area_layer_Source.startswith(u'עיריית תל אביב והסביבה') or
        Tourist_Area_layer_Source.startswith(u'המצפה התת ימי') or
        Tourist_Area_layer_Source.startswith(u'ים סוף - חוף האלמוגים') or
        Tourist_Area_layer_Source.startswith(u'עין בוקק') or
        Tourist_Area_layer_Source.startswith(u'מצוקי דרגות')
           
        ):
                
       return True
    return False

def Update_from_Tourist_Area_layer_Source_str(Activity_Name):
    if (u'כנסיה' in Activity_Name
        or u'כנסייה' in Activity_Name and u'קבר' in Activity_Name):
        return "כנסיית הקבר "
    elif (u'כנסיה' in Activity_Name
        or u'כנסייה' in Activity_Name and u'קבר' in Activity_Name):
        return "כנסייה_"
    if (u'הטבלה' in Activity_Name
        or u'טבילה' in Activity_Name):
        return "הטבלה_"
    if (u'דוידסון' in Activity_Name
        or u'ארכיאולוגי ירושלים' in Activity_Name):
        return "מרכז דוידסון_"
   
    return ""    

def Update_from_Municipal_Area_layer_Source(Activity_Name):
    if (u'בר' in Activity_Name or u'פאב' in Activity_Name
        or u'פלייס' in Activity_Name ):
        return "בר_"
    if (u'טיילת' in Activity_Name or u'טילת' in Activity_Name):
        return "טיילת_"
    if (u'באהיים' in Activity_Name or u'באיים' in Activity_Name or u'באהים' in Activity_Name):
        return "גנים הבאהיים ב"
    if (u'חוף הים' in Activity_Name 
        or u'חוף' in Activity_Name or u'מרינה' in Activity_Name
        ):
        return "חוף הים_"
    if (u'פארק' in Activity_Name ):
        return "פארק_"
    if (u'אופניים' in Activity_Name or u'תל אופן' in Activity_Name):
        return " רכיבה על אופניים"
    if (u'גלריה' in Activity_Name or u'גלרייה' in Activity_Name or u'גלריית' in Activity_Name or u'גלרית' in Activity_Name ):
        return "גלרייה_"
    if (u'צלילה' in Activity_Name ):
        return "צלילה_"
    if ( u'שייט' in Activity_Name or  u'שיט' in Activity_Name):
        return "שייט_"
    if (u'רכבל' in Activity_Name ):
        return "רכבל_"
    if (u'סנימטק' in Activity_Name or u'סינמטק' in Activity_Name ):
        return "סנימטק_"
    if (u'סינימה סיטי' in Activity_Name ):
        return "סינימה סיטי_"
    if (u'הופעה' in Activity_Name ):
        return "הופעה_"
    if (u'בילוי' in Activity_Name
         or u'מסיבה' in Activity_Name
         ):
        return "מועדון_"
    if (u'צדיק' in Activity_Name
         ):
        return "קברי צדיקים_"
    if (u'יקב' in Activity_Name or u'ייקב' in Activity_Name
         ):
        return "יקב_"
    if (u'מטווח' in Activity_Name
         ):
        return "מטווח_"
    if (u'מסלול' in Activity_Name
         ):
        return "מסלול_"
    if (u'הפלגה' in Activity_Name
         ):
        return "הפלגה_"
    if (u'מסגד' in Activity_Name
         ):
        return "מסגד_"
    if (u'קולנוע' in Activity_Name
         ):
        return "קולנוע_"
    if (u'אנדרטה' in Activity_Name
         ):
        return "אנדרטה_"
    if (u'טרקטור' in Activity_Name
         ):
        return "טרקטורונים_"
    if (u'קבלה' in Activity_Name
         ):
        return "מרכז קבלה_"
    if (u'תיאטרון' in Activity_Name
         ):
        return "תיאטרון_"
    if (u'עיר העתיקה' in Activity_Name  ):
        return "העיר העתיקה "
    if (u'נמל' in Activity_Name  ):
        return "נמל "
    if (u'טיול' in Activity_Name or  u'סיור' in Activity_Name or  u'רגלי' in Activity_Name 
        or  u'סיבוב ב' in Activity_Name or u'קניות' in Activity_Name or u'מדרחוב' in Activity_Name ):
        return "מרכז העיר "
    if (u'נחל' in Activity_Name ):
        return "נחל_ "
    if (u'חווה' in Activity_Name ):
        return "חווה_ "
    if (u'בית בד' in Activity_Name or u'שמן זית' in Activity_Name ):
        return "בית בד_ "
    if ('YMCA' in Activity_Name or u'ימקא' in Activity_Name ):
        return "ימקא_ "
    if ('אגם' in Activity_Name  ):
        return "אגם_ "
    if ('באולינג' in Activity_Name  ):
        return "באולינג_ "
    if ('בוטני' in Activity_Name  ):
        return "גן בוטני_ "
    if ('גלישה' in Activity_Name  ):
        return "גלישה_ "
    if ('גולף' in Activity_Name  ):
        return "גולף_ "
   
    return ""

def Change_activity_name(Activity_Name):
    if(u'רובע' in Activity_Name
    or u'שער' in Activity_Name   or u'שאר ה' in Activity_Name 
     or u'דרך דוד' in Activity_Name):
        return "העיר העתיקה ירושלים"
    if (u'ירקון' in Activity_Name):
        return "פארק הירקון"
    if (u'רוטשילד' in Activity_Name):
        return "מרכז העיר תל אביב"
    if(u'כותל' in Activity_Name 
    or  u'כתל' in Activity_Name):
        return "הכותל המערבי"
    if (u'בן גוריון' in Activity_Name):
        return "צריף דוד בן גוריון"
    if (u'מלח' in Activity_Name):
        return "ים המלח"
    if (u'חוף הים_טבריה' in Activity_Name  ):
        return "כנרת"
    if (u'דלרוזה' in Activity_Name or u'ויה ד' in Activity_Name  ):
        return "ויה דלרוזה"
    if (u'נחל פרת' in Activity_Name or u'גן לאומי עין פרת' in Activity_Name  ):
        return "שמורת טבע עין פרת"
    if (u'שרונה' in Activity_Name or u'סרונה' in Activity_Name  ):
        return "מתחם שרונה"
    if (u'ממילא' in Activity_Name  ):
        return "מתחם ממילא"
    if (u'חרמון' in Activity_Name  ):
        return "אתר חרמון"
    if (u'ביד ושם' in Activity_Name  ):
        return "יד ושם"
    if (u'גמל' in Activity_Name  ):
        return "רכיבה על גמלים"
    if (u'מנרה' in Activity_Name  ):
        return "צוק מנרה"
    if (u'אהבה' in Activity_Name  ):
        return "מרכז מבקרים אהבה"
    if (u'נווה צדק' in Activity_Name  ):
        return "נווה צדק"
    if (u'חומות' in Activity_Name  ):
        return "חומות ירושלים"
    if (u'הר הזיתים' in Activity_Name or u'גת' in Activity_Name ):
        return "הר הזיתים"
    if (u'מכתש' in Activity_Name  ):
        return "מכתש רמון"
    if (u'שוק הכרמל' in Activity_Name  ):
        return "שוק הכרמל"
    if (u'שוק הפשפשים' in Activity_Name  ):
        return "שוק הפשפשים"
    if (u'שמעון' in Activity_Name or u'מירון' in Activity_Name or u'רשב' in Activity_Name  ):
        return "קבר הרשבי"
    if (u'תנכי' in Activity_Name   ):
        return "גן החיות התנכי"
    if (u'ירדן' in Activity_Name or u'ירדנית' in Activity_Name  or u'טבילה' in Activity_Name or u'הטבלה' in Activity_Name):
        return "נהר הירדן"
    if (u'אלנבי' in Activity_Name ):
        return "מרכז העיר תל אביב"
    if (u'שוק מחנה' in Activity_Name  ):
        return "שוק מחנה יהודה"
    if (u'יגאל' in Activity_Name  ):
        return "בית יגאל אלון"
    if (u'בן יהודה' in Activity_Name  ):
        return "מרכז העיר ירושלים"
   
    return ""

def Update_from_Tourist_final_area(Activity_Name):
    if (u'תצפית' in Activity_Name):
        return "תצפית_"
    if (u'מנזר' in Activity_Name):
        return "מנזר_"
    if (u'בית עלמין' in Activity_Name):
        return "מנזר_"
   
    return ""

##################################
# importing dataset
##################################
df  = pd.read_csv("G:\\My Drive\\practicum\\individual tourist activities.csv")
print(df.describe())
print(df.head())
print(df.shape)

######cleaning####
drop_list=[]
for i, row in df.iterrows():
    t=Unneeded_Row(row['Activity_Name'],
                   row['activity_purpose'], 
                   row['means_of_transportation'],
                   row['visit_purpose_category'],
                   row['country_area'],)
    if t==True:
       drop_list.append(i)
#########סידור מילים###
    ins=str(row['Tourist_Area_layer_Source'])
    fin=str(row['tourist_final_aria'])
   
    t=Update_from_Tourist_Area_layer_Source(ins)
    if t==True:
       df.at[i, 'Activity_Name'] = row['Tourist_Area_layer_Source']

    s=Update_from_Municipal_Area_layer_Source(str(row['Activity_Name']))
    if s!='':
        if (str(row['Municipal_Area_layer_Source'])!='-1'):
           df.at[i, 'Activity_Name'] =s+str(row['Municipal_Area_layer_Source'])
        else:
           df.at[i, 'Activity_Name'] = s
   
    z=Update_from_Tourist_final_area(row['Activity_Name'] )
    if z!='':
        if (str(row['tourist_final_aria'])!='-1'):
           df.at[i, 'Activity_Name'] = z+str(row['tourist_final_aria'])
        else:
           df.at[i, 'Activity_Name'] = z
       
    x=Update_from_Tourist_Area_layer_Source_str(row['Activity_Name'])    
    if x!='':
           if (str(row['Tourist_Area_layer_Source'])!='-1'):
              df.at[i, 'Activity_Name'] = x+str(row['Tourist_Area_layer_Source'])
           else:
              df.at[i, 'Activity_Name'] = x
         
    y=Change_activity_name(row['Activity_Name'])    
    if y!='':
        df.at[i, 'Activity_Name'] = y
              
    
print('Rows dropped',len(drop_list))
df = df.drop(drop_list)        

print('Cleaned')

###feature selection####

df=df.iloc[:,[0,4,8,11,12,13,14,15,16,17,18,64,66]]
activities=df.copy()
print(df.describe())
print(df.head())
print(df.shape)

df.to_excel("activities_version1.xlsx") 
 

#Visualizations
#Preparing data for visualization purposes
individuals  = pd.read_excel("G:\\My Drive\\practicum\\individuals_v1.xlsx")
print(individuals.describe())
print(individuals.head())

# =============================================================================
# individuals = individuals.reset_index().set_index('Record_Id')
# individuals['Record_Id'] = individuals.index
# =============================================================================

df['Activity_Name']=df['Activity_Name'][::-1]
df['number_of_groups'] = 1
df['number_of_activities']=1

df['t_age_0_12']=0
df['t_age_13_18']=0
df['t_age_19_24']=0
df['t_age_25_39']=0
df['t_age_40_59']=0
df['t_age_60_74']=0
df['t_age_75plus']=0
  
for i, row in df.iterrows():        
    a=individuals.loc[individuals['Record_Id']==row['Record_Id']].head(1) 
    df.at[i, 'visit_purpose_category'] = a['purpose'].max()
    df.at[i, 'Country_of_Residence'] = a['country'].max()
    df.at[i, 'country_of_origin_category'] = a['country_category'].max()
    df.at[i, 'Summer_Winter'] = a['Summer_Winter'].max()
    df.at[i, 'first_visit'] = a['first_visit'].max()
    df.at[i, 'main_purpose'] = a['purpose'].max()
    
    df.at[i, 'How_many_people_in_group'] = a['group_size'].max()   
    df.at[i, 'Number_of_women'] = a['women_in_group'].max()
    df.at[i, 'Number_of_men'] = a['men_in_group'].max()
    df.at[i, 'number_of_nights'] = a['number_of_nights'].max() 
 
    
    df.at[i, 't_age_0_12'] =a['male_age_0_12']+a['female_age_0_12']
    df.at[i, 't_age_13_18']=a['male_age_13_18']+a['female_age_13_18']
    df.at[i, 't_age_19_24']=a['male_age_19_24']+a['female_age_19_24']
    df.at[i, 't_age_25_39']=a['male_age_25_39']+a['female_age_25_39']
    df.at[i, 't_age_40_59']=a['male_age_40_59']+a['female_age_40_59']
    df.at[i, 't_age_60_74']=a['male_age_60_74']+a['female_age_60_74']
    df.at[i, 't_age_75plus']=a['male_age_75plus']+a['female_age_75plus']
        
df.to_excel("combined.xlsx") 

print('Finished')

#Visualizations:
def plot_top_n(x, y, title, n=10):
    count = {}
    for i in range(len(x)):
        if y[i] in count:
            count[y[i]] += 1
        else:
            count[y[i]] = 1
            
    total = sum(count.values())
    
    sorted_count = dict(sorted(count.items(), key=lambda item: item[1], reverse=True)[:n])
    
    percents = {k: round(v/total*100, 1) for k, v in sorted_count.items()}
     
    plt.bar(range(len(sorted_count)), list(sorted_count.values()), align='center')
    for i, v in enumerate(list(sorted_count.values())):
        plt.text(i, v+0.1, f"{percents[list(percents.keys())[i]]}%", ha='center')
    plt.xticks(range(len(sorted_count)), list(percents.keys()), rotation='vertical') 
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title(title)
    plt.show()
  
def plot_top_n_percent(x, y, title, n=10):

    groups = set(x)
    counts = {group: {} for group in groups}

    for i in range(len(x)):
        group = x[i]
        instance = y[i]
        if instance in counts[group]:
            counts[group][instance] += 1
        else:
            counts[group][instance] = 1

    for group in groups:
        total = sum(counts[group].values())
        for instance in counts[group]:
            counts[group][instance] /= total

    sorted_counts = {group: dict(sorted(counts[group].items(), key=lambda x: x[1], reverse=True)[:n]) for group in groups}

    ind = list(range(n))
    width = 0.8/len(groups)

    for i, group in enumerate(sorted_counts):
        plt.bar(ind, list(sorted_counts[group].values()), width, label=group)
        ind = [x + width for x in ind]
        
    plt.xticks(range(n), list(sorted_counts[group].keys()), rotation='vertical')
    plt.xlabel("Category")
    plt.ylabel("Percentage")
    plt.title(title)
    plt.legend()
    plt.show()
      
def plot_top_n_gender(x, y, title, n=5):
    
    men_y = [y[i] for i in range(len(x)) if x[i] != 0]
    women_y = [y[i] for i in range(len(x)) if x[i] == 0]

    men_count = {}
    women_count = {}

    for i in men_y:
        men_count[i] = men_count.get(i, 0) + 1

    for i in women_y:
        women_count[i] = women_count.get(i, 0) + 1

    men_top = dict(sorted(men_count.items(), key=lambda x: x[1], reverse=True)[:n])
    women_top = dict(sorted(women_count.items(), key=lambda x: x[1], reverse=True)[:n])

    men_perc = {k: v / len(men_y) for k, v in men_top.items()}
    women_perc = {k: v / len(women_y) for k, v in women_top.items()}

    # Combine all unique categories from men_perc and women_perc
    all_categories = list(set(list(men_perc.keys()) + list(women_perc.keys())))

    # Sort the combined categories alphabetically
    all_categories.sort()

    # Generate an array of indices for the x-axis ticks
    ind = np.arange(len(all_categories))

    # Set the width of the bars
    width = 0.4

    # Create an array with zeros representing the men's percentages
    men_values = np.zeros(len(all_categories))
    for i, cat in enumerate(all_categories):
        men_values[i] = men_perc.get(cat, 0)

    # Create an array with zeros representing the women's percentages
    women_values = np.zeros(len(all_categories))
    for i, cat in enumerate(all_categories):
        women_values[i] = women_perc.get(cat, 0)

    # Plot the bar chart
    p1 = plt.bar(ind, men_values, width)
    p2 = plt.bar(ind + width, women_values, width)
    
    plt.ylabel('Percentage')
    plt.title(title)
    plt.xticks(ind + width / 2, all_categories, rotation ='vertical')
    plt.legend((p1[0], p2[0]), ('Men', 'Women'))

    plt.show()
  
def plot_top_n_group(x, y, title, n_groups=5, n_values=5):

    groups = set(x) 
    group_count = {g:{} for g in groups}

    for i in range(len(x)):
        group = x[i]
        value = y[i]
        group_count[group][value] = group_count[group].get(value, 0) + 1

    top_groups = dict(sorted(group_count.items(), key=lambda x: sum(x[1].values()), reverse=True)[:n_groups])

    for group in top_groups:
        total = sum(top_groups[group].values())
        for value in top_groups[group]:
            top_groups[group][value] /= total
        top_groups[group] = dict(sorted(top_groups[group].items(), key=lambda x: x[1], reverse=True)[:n_values])

    indexes = np.arange(n_values)
    width = 0.8/len(top_groups)

    for i, group in enumerate(top_groups):
        plt.bar(indexes + i*width, top_groups[group].values(), width, label=group)

    plt.xticks(indexes+width, top_groups[group].keys(),rotation ='vertical')
    plt.legend()
    plt.title(title)
    plt.show()

def check_stay_vs_activities(df, days_col, activities_col):

  df['num_activities'] = df.groupby([days_col])[activities_col].transform('nunique')

  plt.scatter(df[days_col], df['num_activities'])

  plt.xlabel('Length of Stay (Days)')
  plt.ylabel('Number of Unique Activities')

  plt.title('Relationship Between Length of Stay and Activities')

  ma

  plt.show()
  
def plot_top_activities_by_age(df, age_col, activity_col, category_col, n=5):

  df_subset = df[[age_col, activity_col, category_col]]  

  categories = df_subset[category_col].unique()

  for cat in categories:

    df_cat = df_subset[df_subset[category_col] == cat]
    
    age_group = age_col # or lookup name from mapping

    df_age = df_cat[df_cat[age_col] == df_cat[age_col]]

    top = df_age[activity_col].value_counts().nlargest(n)

    plt.bar(top.index, top.values)
    plt.xticks(rotation=90)
    plt.title(f'Top {n} Activities for Age Group {age_group} and Category {cat}')
    plt.ylabel('Frequency')
    plt.xlabel('Activity')
    plt.show()
##################################################################   
print(df.describe())
print(df.head())

df["Activity_Name"]=df["Activity_Name"].apply(lambda x: x[::-1])

uniqe=df["Activity_Name"].nunique()

#####################
def df_to_series(ser):
    ser=ser.reset_index()
    ser.drop(['index'], axis=1, inplace=True)
    ser=ser.squeeze()
    return ser
    

y=df["Activity_Name"]
y=df_to_series(y)
x=df["number_of_groups"]
title="top 10 sites for groups"
general_frequency=plot_top_n(x, y, title, n=10)
x=df["How_many_people_in_group"]
title="top 10 sites for amount of pepole"
general_frequency=plot_top_n(x, y, title, n=10)

howmany=df["How_many_people_in_group"].describe()

###conclusion ; exactly the same

#####visit_purpose_category
df_visit_purpose_category=df.loc[:,['Activity_Name','visit_purpose_category','number_of_groups']]

y=df_visit_purpose_category["visit_purpose_category"]
y=df_to_series(y)
x=df_visit_purpose_category["number_of_groups"]
title="visit_purpose_category"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_visit_purpose_category["Activity_Name"]
y=df_to_series(y)
x=df_visit_purpose_category["visit_purpose_category"]
x=df_to_series(x)
title="top 10 sites for visit_purpose_category"
plot_top_n_percent(x, y, title, n=10)

#############religious_affiliation_category
df_religion_category=df.loc[:,['Activity_Name','religious_affiliation_category','number_of_groups']]

y=df_religion_category["religious_affiliation_category"]
y=df_to_series(y)
x=df_religion_category["number_of_groups"]
title="religious_affiliation_category"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_religion_category["Activity_Name"]
y=df_to_series(y)
x=df_religion_category["religious_affiliation_category"]
x=df_to_series(x)
title="top 10 sites for religious_affiliation_category"
plot_top_n_percent(x, y, title, n=10)

##########first_visit
df_first_visit=df.loc[:,['Activity_Name','first_visit','number_of_groups']]

y=df_first_visit["first_visit"]
y=df_to_series(y)
x=df_first_visit["number_of_groups"]
title="first_visit"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_first_visit["Activity_Name"]
y=df_to_series(y)
x=df_first_visit["first_visit"]
x=df_to_series(x)
title="top 10 sites for first_visit"
plot_top_n_percent(x, y, title, n=10)

###Number_of_people_in_group_category

df_Number_of_people_in_group_category=df.loc[:,['Activity_Name','Number_of_people_in_group_category','number_of_groups']]

y=df_Number_of_people_in_group_category["Number_of_people_in_group_category"]
y=df_to_series(y)
x=df_Number_of_people_in_group_category["number_of_groups"]
title="Number_of_people_in_group_category"
general_frequency=plot_top_n(x, y, title, n=10)


y=df_Number_of_people_in_group_category["Activity_Name"]
y=df_to_series(y)
x=df_Number_of_people_in_group_category["Number_of_people_in_group_category"]
x=df_to_series(x)
title="top 10 sites for Number_of_people_in_group_category"
plot_top_n_percent(x, y, title, n=10)

####length_of_visit
df_length_of_visit=df.loc[:,['Activity_Name','length_of_visit','number_of_groups']]

y=df_length_of_visit["length_of_visit"]
y=df_to_series(y)
x=df_length_of_visit["number_of_groups"]
title="length_of_visit"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_length_of_visit["Activity_Name"]
y=df_to_series(y)
x=df_length_of_visit["length_of_visit"]
x=df_to_series(x)
title="top 10 sites for length_of_visit"
plot_top_n_percent(x, y, title, n=10)

days_col="number_of_nights"
activities_col="Activity_Name"
check_stay_vs_activities(df, days_col, activities_col)

#####Summer_Winter
df_Summer_Winter=df.loc[:,['Activity_Name','Summer_Winter','number_of_groups']]

y=df_Summer_Winter["Summer_Winter"]
y=df_to_series(y)
x=df_Summer_Winter["number_of_groups"]
title="Summer_Winter"
general_frequency=plot_top_n(x, y, title, n=10)


y=df_Summer_Winter["Activity_Name"]
y=df_to_series(y)
x=df_Summer_Winter["Summer_Winter"]
x=df_to_series(x)
title="top 10 sites forSummer_Winter"
plot_top_n_percent(x, y, title, n=10)

############arrival_season
df_arrival_season=df.loc[:,['Activity_Name','arrival_season','number_of_groups']]

y=df_arrival_season["arrival_season"]
y=df_to_series(y)
x=df_arrival_season["number_of_groups"]
title="arrival_season"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_arrival_season["Activity_Name"]
y=df_to_series(y)
x=df_arrival_season["arrival_season"]
x=df_to_series(x)
title="top 10 sites for arrival_season"
plot_top_n_percent(x, y, title, n=10)

#####country_area
df_country_area=df.loc[:,['Activity_Name','country_area','number_of_groups']]

y=df_country_area["country_area"]
y=df_to_series(y)
x=df_country_area["number_of_groups"]
title="country_area"
general_frequency=plot_top_n(x, y, title, n=10)

#####Number_of_men---- binary 
df_Number_of_men=df.loc[:,['Activity_Name','Number_of_men','number_of_groups']]
manstas=df_Number_of_men.sum()

y=df_Number_of_men["Activity_Name"]
y=df_to_series(y)
x=df_Number_of_men["Number_of_men"]
x=df_to_series(x)
title="top 5 sites for men and women count"
gender_plt=plot_top_n_gender(x, y, title, n=5)

############Country of Residence
df_country_of_residence=df.loc[:,['Activity_Name','Country_of_Residence','number_of_groups']]

y=df_country_of_residence["Country_of_Residence"]
y=df_to_series(y)
x=df_country_of_residence["number_of_groups"]
title="country_of_residence"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_country_of_residence["Activity_Name"]
y=df_to_series(y)
x=df_country_of_residence["Country_of_Residence"]
x=df_to_series(x)
title="top 5 sites for top 5 country of Residence"
plot_top_n_group(x, y, title, n_groups=5, n_values=5)

###country of origin
df_country_of_origin_category=df.loc[:,['Activity_Name','country_of_origin_category','number_of_groups']]

y=df_country_of_origin_category["country_of_origin_category"]
y=df_to_series(y)
x=df_country_of_origin_category["number_of_groups"]
x=df_to_series(x)
title="country_of_origin_category"
general_frequency=plot_top_n(x, y, title, n=10)

y=df_country_of_origin_category["Activity_Name"]
y=df_to_series(y)
x=df_country_of_origin_category["country_of_origin_category"]
x=df_to_series(x)
title="top 5 sites for top 5 country_of_origin_category"
plot_top_n_group(x, y, title, n_groups=5, n_values=5)

####age groups
df_age_groups=df.loc[:,['Activity_Name','t_age_0_12','t_age_13_18','t_age_19_24','t_age_25_39','t_age_40_59','t_age_60_74','t_age_75plus']]
age_groups_sum=df_age_groups.sum()
sum_age=age_groups_sum[1:].sum()
precent_age=((age_groups_sum[1:])/sum_age)*100

plt.bar(precent_age.index, precent_age.values)
plt.gca().set_xticklabels(precent_age.index, rotation=90)
plt.ylabel("Percentage")
plt.title("Age Group Percentages")

# Add percentage labels above bars  
for i, v in enumerate(precent_age):
    plt.text(i, v+0.5, str(round(v, 1)) + '%', ha='center')
    
plt.show()

age_col="t_age_25_39"
activity_col= "Activity_Name"
category_col= "first_visit"

plot_top_activities_by_age(df ,age_col ,activity_col, category_col, n=5)

age_col="t_age_40_59"
plot_top_activities_by_age(df ,age_col ,activity_col, category_col, n=5)

#Total number of rows in the dataframe

rec_id=individuals.loc[:,'Record_Id']
total=len(rec_id)
# pupose vs religion
    
plt.figure(figsize=(15,10))
religion_purpose=sns.countplot(data=individuals,x='purpose',hue='religion')
for p in religion_purpose.patches:
    height=p.get_height()
    religion_purpose.text(p.get_x()+p.get_width()/2.,height+10,'{:1.0f}%'.format(height/total*100),ha='center')
religion_purpose.set_title("purpose by religion")
religion_purpose.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.figure()

#israeli passport-y/n
plt.figure(figsize=(15,10))
israeli_pass=sns.countplot(data=individuals,x='Israeli_passport',hue='Israeli_passport')
for p in israeli_pass .patches:
    height=p.get_height()
    israeli_pass.text(p.get_x()+p.get_width()/2.,height+100,'{:1.0f}%'.format(height/total*100),ha='center')
israeli_pass.set_title("Israeli passport- yes/no")
israeli_pass.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.figure()

#number of women / men in group

print(individuals[["women_in_group","men_in_group"]].describe(include="all"))

plt.figure(figsize=(20,8))

bins_w_m_num=[0,1,2,3,8.5]
labels_w_m_num=['0','1','2','3+']  
        
plt.subplot(1,2,1)
individuals['bins_w_num']=pd.cut(individuals['women_in_group'], bins=bins_w_m_num, labels=labels_w_m_num,include_lowest=True,right=False)

w_num=sns.countplot(data=individuals,x='bins_w_num',hue='bins_w_num')
for p in w_num .patches:
    height=p.get_height()
    w_num.text(p.get_x()+p.get_width()/2.,height+2,'{:1.0f}%'.format(height/total*100),ha='center')
w_num.set_title("number of women in group")
w_num.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.subplot(1,2,2)
individuals['bins_m_num']=pd.cut(individuals['men_in_group'], bins=bins_w_m_num, labels=labels_w_m_num,include_lowest=True,right=False)
m_num=sns.countplot(data=individuals,x='bins_m_num',hue='bins_m_num')
for p in m_num .patches:
    height=p.get_height()
    m_num.text(p.get_x()+p.get_width()/2.,height+2,'{:1.0f}%'.format(height/total*100),ha='center')
m_num.set_title("number of men in group")
m_num.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

plt.figure()

# number of nights vs purpose
#number of nights

bins_nights=[0,4,7,10,16.5]
labels_nights=['0-3','4-6','7-10','10+']

individuals['bins_nights_num']=pd.cut(individuals['number_of_nights'], bins=bins_nights, labels=labels_nights,include_lowest=True,right=False)

plt.figure(figsize=(15,10))
nights_purpose=sns.countplot(data=individuals,x='purpose',hue='bins_nights_num')
for p in nights_purpose.patches:
    height=p.get_height()
    nights_purpose.text(p.get_x()+p.get_width()/2.,height+10,'{:1.0f}%'.format(height/total*100),ha='center')
nights_purpose.set_title("Visit's purpose by number of nights")
nights_purpose.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,title='number of nights')
plt.figure()
#Create a dictionary with key = tourist id , value = list of activities that he did  

ind_act_dict={}
for i in activities.index:  
    if activities['Record_Id'][i] in ind_act_dict:
        ind_act_dict[activities['Record_Id'][i]].append(activities['Activity_Name'][i])
    else:
        ind_act_dict[activities['Record_Id'][i]]=[activities['Activity_Name'][i]]

#Adding activities columns to individuals dataframe:        
activity_dict={}
j=1
for i in activities['Activity_Name']:
    if i not in activity_dict:
        activity_dict[i]=j
        j+=1
#Create a dictionary with key = tourist id and value = list of activities that he did     

ind_act_dict={}
for i in activities.index:  
    if activities['Record_Id'][i] in ind_act_dict:
        ind_act_dict[activities['Record_Id'][i]].append(activities['Activity_Name'][i])
    else:
        ind_act_dict[activities['Record_Id'][i]]=[activities['Activity_Name'][i]]
  #Creates a column for each activity name with all 0 as a start
  
for key in activity_dict:
    individuals[key]=0
#In each activity column, change value to 1 if the person participated in the activity
    
for i in individuals.index:
    if individuals['Record_Id'][i] in ind_act_dict:
        for num in ind_act_dict[individuals['Record_Id'][i]]:
            individuals[num][i]=1


individuals.to_excel("individuals_with_binary_activities.xlsx" ,index=False)
          
# Create a new excel file with the binary activity columns
        
def create_basic_dict(og_dict,ls):
    d=dict([(key, og_dict[key]) for key in ls])
    return d

def create_freq_dict(og_dict,ls):
    d=create_basic_dict(og_dict, ls)
    frq_d={}
    for l in d.values():
        for i in l:
            if i in frq_d:
                frq_d[i]+=1
            else:
                frq_d[i]=1
    return frq_d

def create_frq_df(og_dict,ls,col1):
    frq_d=create_freq_dict(og_dict, ls)
    frq_df=pd.DataFrame.from_dict(frq_d,orient='index').reset_index()
    frq_df.columns = [col1,'frequency']
    frq_df=frq_df.sort_values(by=['frequency'], ascending=False)
    frq_df[col1] = frq_df.loc[:,col1].apply(lambda x: x[::-1])
    return frq_df

def filter_rows_by_values(df, col, values):
    df= df[~df[col].isin(values)]
    return df

#Compare id list in individuals vs activities

individuals_id_ls=individuals['Record_Id'].tolist()
activities_id_ls=list(ind_act_dict.keys())
id_differences=list(set(individuals_id_ls) - set(activities_id_ls))
# Remove id_differences from datasets
individuals=filter_rows_by_values(individuals, 'Record_Id', id_differences)
activities=filter_rows_by_values(activities, 'Record_Id', id_differences)

individuals_updated_id_ls=individuals['Record_Id'].tolist()

full_sites_freq_df=create_frq_df(ind_act_dict, individuals_updated_id_ls, 'site')
sites_further_analysis=full_sites_freq_df.copy() 
sites_further_analysis=sites_further_analysis[sites_further_analysis['frequency'] > 10] 
sites_further_analysis['site'] = sites_further_analysis.loc[:,'site'].apply(lambda x: x[::-1])

#Create site list for unsupervised modeling 

id_vec_dict={}
for i in individuals.index: 
    vec=[]
    id_vec_dict[individuals['Record_Id'][i]]=vec
    for j in sites_further_analysis['site']:
        vec.append(individuals[j][i])


#Statistical testing to support our essumptions

id_vec_1={}
for k,v in id_vec_dict.items():
    sum_1=0
    for num in v:
        if num==1:
            sum_1+=num
    id_vec_1[k]=sum_1
    
individuals['id_vec_1']=individuals['Record_Id'].map(id_vec_1)

def create_dict_vec1(col,df):
    my_dict={}
    for i in df.index:
        if df[col][i] in my_dict:
            my_dict[df[col][i]].append(df['id_vec_1'][i])
        else:
            my_dict[df[col][i]]=[df['id_vec_1'][i]]
    return my_dict
    
rel1=[]
for i in individuals.index:
    if individuals['religion'][i]=='Christian':
        rel1.append('Christian')
    elif individuals['religion'][i]=='Jewish - Other':
        rel1.append('Jewish - Other')
    elif individuals['religion'][i]=='Jewish - Religious':
        rel1.append('Jewish - Religious')
    else:
        rel1.append('other')

individuals['religion_1']=rel1    
    
rel_vec_1=create_dict_vec1('religion_1', individuals)

#Create a site vector for each tourist id

df1=individuals[['Record_Id','country']]
n = 5
top5countries=individuals['country'].value_counts()[:n].index.tolist()
df1=df1[df1['country'].isin(top5countries)]
df1['id_vec_1']=df1['Record_Id'].map(id_vec_1)
    
print("Results for Anova test- religion vs. activities:")
f_statistic_rel, p_value_rel =f_oneway(rel_vec_1['Christian'],rel_vec_1['Jewish - Other'],rel_vec_1['Jewish - Religious'],rel_vec_1['other'])            
print(f"F-statistic: {f_statistic_rel}")
print(f"p-value: {p_value_rel}")
print()
#--------------------------
df1['id_vec_1']=df1['Record_Id'].map(id_vec_1)

country_vec_1=create_dict_vec1('country', df1)

print("Results for Anova test- country of origin vs. activities:")
f_statistic_country, p_value_country =f_oneway(country_vec_1['United States'],country_vec_1['United Kingdom'],country_vec_1['Germany'],country_vec_1['France'],country_vec_1['Netherlands'])            
print(f"F-statistic: {f_statistic_country}")
print(f"p-value: {p_value_country}")
print()
#--------------------------
individuals['group_bins']=individuals['group_size']
individuals['group_bins'].loc[individuals['group_size']==1] = 'sole tourist'
individuals['group_bins'].loc[individuals['group_size']==2] = 'couples'
individuals['group_bins'].loc[individuals['group_size']>2] = '3+'

gsize_vec_1=create_dict_vec1('group_bins', individuals)
        
print("Results for Anova test- group size bins vs. activities:")
f_statistic_gsize, p_value_gsize =f_oneway(gsize_vec_1['sole tourist'],gsize_vec_1['couples'],gsize_vec_1['3+'])           
print(f"F-statistic: {f_statistic_gsize}")
print(f"p-value: {p_value_gsize}")
print()
#----------------------------
individuals['sole_tourist'] = individuals['group_size']
individuals['sole_tourist'].loc[individuals['group_size']==1] = 'yes'
individuals['sole_tourist'].loc[individuals['group_size']>1] = 'no'

sole_vec_1=create_dict_vec1('sole_tourist', individuals)

ttest,pval_sole = ttest_ind(sole_vec_1['yes'],sole_vec_1['no'])

print("Results for t-test- alone-y/n vs. activities:")
if pval_sole <0.05:
  print("\nREJECTING THE NULL HYPOTHESIS")
else:
  print("\nACCEPTING THE NULL HYPOTHESIS")

print()
#---------------------------  
individuals['short/long_visit'] = individuals['number_of_nights']
individuals['short/long_visit'].loc[individuals['number_of_nights']<=3] = 'short'
individuals['short/long_visit'].loc[individuals['number_of_nights']>3] = 'long'

sl_vec_1=create_dict_vec1('short/long_visit', individuals)

ttest,pval_sl = ttest_ind(sl_vec_1['short'],sl_vec_1['long'])

print("Results for t-test- short/long visit vs. activities:")
if pval_sl <0.05:
  print("\nREJECTING THE NULL HYPOTHESIS")
else:
  print("\nACCEPTING THE NULL HYPOTHESIS")

print()
#----------------------------

first_vec_1=create_dict_vec1('first_visit', individuals)

ttest,pval_first = ttest_ind(first_vec_1['Yes'],first_vec_1['No'])

print("Results for t-test- first visit-y/n vs. activities vector:")
if pval_sl <0.05:
  print("\nREJECTING THE NULL HYPOTHESIS")
else:
  print("\nACCEPTING THE NULL HYPOTHESIS")
  
print()

rec_age_df=individuals.loc[:,["male_age_0_12","male_age_13_18","male_age_19_24","male_age_25_39","male_age_40_59","male_age_60_74","male_age_75plus",'female_age_0_12','female_age_13_18','female_age_19_24','female_age_25_39','female_age_40_59','female_age_60_74','female_age_75plus']]
rec_age_df=rec_age_df.rename(columns={"male_age_0_12": "1","male_age_13_18": "2","male_age_19_24": "3","male_age_25_39": "4","male_age_40_59": "5","male_age_60_74": "6","male_age_75plus": "7",'female_age_0_12':'a','female_age_13_18':'b','female_age_19_24':'c','female_age_25_39':'d','female_age_40_59':'e','female_age_60_74':'f','female_age_75plus':'g'})
age_dict={1:"0_12",2:"13_18",3:"19_59",4:"60_74",5:"75+"}
tourist_age_young={}
tourist_age_old={}

for i in rec_age_df.index:
    if rec_age_df['1'][i]==1 or rec_age_df['a'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[1]
        continue
    elif rec_age_df['7'][i]==1 or rec_age_df['g'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[5]
        continue
    elif rec_age_df['2'][i]==1 or rec_age_df['b'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[2]
        continue
    elif rec_age_df['6'][i]==1 or rec_age_df['f'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[4]
        continue
    else:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[3]

for i in rec_age_df.index:

    if rec_age_df['7'][i]==1 or rec_age_df['g'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[5]
        continue
    elif rec_age_df['1'][i]==1 or rec_age_df['a'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[1]
        continue
    elif rec_age_df['6'][i]==1 or rec_age_df['f'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[4]
        continue
    elif rec_age_df['2'][i]==1 or rec_age_df['b'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[2]
        continue
    else:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[3]
        
#old=75+ (oldest) / 60-74 (second)
#young=0_12 (youngest) / 13_18 (second)
#Creating an age tagging dictionary for each tourist that emplify the nost "extreme" individual (age wise), within a record.
#tourist_age_young gives more focus to the younger individual (if exist) while tourist_age_old gives more focus to the older individual (if exist).

#Adding the age tagging to individuals dataset
individuals['age_tag_y']=individuals['Record_Id'].map(tourist_age_young)
individuals['age_tag_o']=individuals['Record_Id'].map(tourist_age_old)


age_y_vec_1=create_dict_vec1('age_tag_y', individuals)
print("Results for Anova test- tourist age group (emphasis=young) vs. activities:")
f_statistic_age_y, p_value_age_y =f_oneway(age_y_vec_1['0_12'],age_y_vec_1['13_18'],age_y_vec_1['19_59'],age_y_vec_1['60_74'],age_y_vec_1['75+'])           
print(f"F-statistic: {f_statistic_age_y}")
print(f"p-value: {p_value_age_y}")
print()

age_o_vec_1=create_dict_vec1('age_tag_o', individuals)
print("Results for Anova test- tourist age group (emphasis=old) vs. activities:")
f_statistic_age_o, p_value_age_o =f_oneway(age_o_vec_1['0_12'],age_o_vec_1['13_18'],age_o_vec_1['19_59'],age_o_vec_1['60_74'],age_o_vec_1['75+'])           
print(f"F-statistic: {f_statistic_age_o}")
print(f"p-value: {p_value_age_o}")
print()




#################################################
######################models#####################
#################################################

#####Frequent itemset and rule mining########

'''Explanitions

Support: measures the number of times a particular item or combination of items occur in a dataset.
Confidence: It measures how the consumer is likely to consume x given they have consumed y.
Lift: The strength of association between the best rules. It is obtained by taking confidence and diving it with support.
 
'''
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
from mlxtend.frequent_patterns import association_rules



#Rule mining visualization
 # Function to convert rules to coordinates.
def rules_to_coordinates(rules):
     antecedents= rules['antecedents'].apply(lambda antecedents: list(antecedents)[0])
     antecedents= antecedents.apply(lambda x: x[::-1])
     consequent = rules['consequents'].apply(lambda consequents: list(consequents)[0])
     consequent = consequent.apply(lambda x: x[::-1])
     rules['rule'] = rules.index
     
     df=pd.concat([antecedents,consequent,rules['rule']],axis=1)
     
     return df


def frequent_itemsets_mlxtend(df, min_support, min_confidence, lift):
  
    # Perform frequent itemset mining using the Apriori algorithm from mlxtend
    frequent_itemsets = apriori(df.astype('bool'), min_support=min_support, use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_larger_than1=frequent_itemsets[ (frequent_itemsets['length'] >1)] 
                     # &(frequent_itemsets['support'] >= 0.15) ]
    print(frequent_itemsets_larger_than1["itemsets"])
    
    # Generate association rules from the frequent itemsets
    rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
 
    if lift==True:   
 # Sort the rules by lift in descending order
        rules = rules.sort_values(by="lift", ascending=False)
        rules=rules[ (rules['lift'] >1)] 
        for i,rule in rules.iterrows():
            print("apriori positive rules:")
            print(str(rule["antecedents"])+">>>"+(str(rule["consequents"])))
            print(" ")
        
    from pandas.plotting import parallel_coordinates
         # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules)
  
            # Generate parallel coordinates plot
    plt.figure(figsize=(4,8))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.show()

    import networkx as nx
    fig, ax=plt.subplots(figsize=(10,4))
    GA=nx.from_pandas_edgelist(coords,source='antecedents',target='consequents')
    nx.draw(GA,with_labels=True)
    plt.show()
   

    return rules

def fpmax_asocsiation_rules(df, min_support, min_confidence,lift):
    
    # Perform frequent itemset mining using the Apriori algorithm from mlxtend
    frequent_itemsets = fpmax(df.astype('bool'), min_support=min_support , use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_larger_than1=frequent_itemsets[ (frequent_itemsets['length'] >1)] 
                     # &(frequent_itemsets['support'] >= 0.15) ]
    print(frequent_itemsets_larger_than1["itemsets"])
    
    # Generate association rules from the frequent itemsets
    rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, support_only=True)
 
   
    if lift==True:   
 # Sort the rules by lift in descending order
        rules = rules.sort_values(by="lift", ascending=False)
        rules=rules[ (rules['lift'] >1)] 
        for i,rule in rules.iterrows():
            print("fpmax positive rules:")
            print(str(rule["antecedents"])+">>>"+(str(rule["consequents"])))
            print(" ")
    from pandas.plotting import parallel_coordinates
             # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules)
      
                # Generate parallel coordinates plot
    plt.figure(figsize=(4,8))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.show()

    import networkx as nx
    fig, ax=plt.subplots(figsize=(10,4))
    GA=nx.from_pandas_edgelist(coords,source='antecedents',target='consequents')
    nx.draw(GA,with_labels=True)
    plt.show()
    
    

    return rules


def fpgrowth_asocsiation_rules(df, min_support, min_confidence,lift):
    
    # Perform frequent itemset mining using the Apriori algorithm from mlxtend
    frequent_itemsets = fpgrowth(df.astype('bool'), min_support=min_support , use_colnames=True)
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    frequent_itemsets_larger_than1=frequent_itemsets[ (frequent_itemsets['length'] >1)] 
                     # &(frequent_itemsets['support'] >= 0.15) ]
    print(frequent_itemsets_larger_than1["itemsets"])
    
    # Generate association rules from the frequent itemsets
    rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
 
    if lift==True:   
# Sort the rules by lift in descending order
       rules = rules.sort_values(by="lift", ascending=False)
       rules=rules[ (rules['lift'] >1)] 
       for i,rule in rules.iterrows():
           print("fpgrowth positive rules:")
           print(str(rule["antecedents"])+">>>"+(str(rule["consequents"])))
           print(" ")
    from pandas.plotting import parallel_coordinates
             # Convert rules into coordinates suitable for use in a parallel coordinates plot
    coords = rules_to_coordinates(rules)
      
                # Generate parallel coordinates plot
    plt.figure(figsize=(4,8))
    parallel_coordinates(coords, 'rule')
    plt.legend([])
    plt.grid(True)
    plt.show()

    import networkx as nx
    fig, ax=plt.subplots(figsize=(10,4))
    GA=nx.from_pandas_edgelist(coords,source='antecedents',target='consequents')
    nx.draw(GA,with_labels=True)
    plt.show()
    

    return rules




#create df for rule minig model
removed_site_list=[]
general_site_list=individuals.iloc[:,97:-2].columns.values.tolist()#all sites list
finished_site_list=sites_further_analysis['site'].values.tolist()#filterd site list

finished_site_list_rule_mining=finished_site_list[5:-1]#site liste, filterd, without top 5
for site in general_site_list:
    if site not in finished_site_list_rule_mining:
        removed_site_list.append(site)
site_df=individuals.iloc[:,97:-2]
for site in site_df.columns.values:
       if site in removed_site_list:
           site_df=site_df.drop([site], axis=1)# filted site df

min_support = 0.05
min_confidence = 0.01
rules = frequent_itemsets_mlxtend(site_df, min_support, min_confidence, lift=True)

########################fpmax asociation rules
min_support=0.05
min_confidence=0.1
rules=fpmax_asocsiation_rules( site_df, min_support,min_confidence, lift=True  )
rules=fpmax_asocsiation_rules( site_df, min_support,min_confidence, lift=False  )

##################################fpgrowth
min_support=0.05
min_confidence=0.2
rules=fpgrowth_asocsiation_rules(site_df, min_support, min_confidence, lift=True)

#Clustering

#K-Means Clustering 

vec_ls = list(id_vec_dict.values())
vec_ls_np=np.array(vec_ls)

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)
 
for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(vec_ls_np)
    kmeanModel.fit(vec_ls_np)
 
    distortions.append(sum(np.min(cdist(vec_ls_np, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / vec_ls_np.shape[0])
    inertias.append(kmeanModel.inertia_)
 
    mapping1[k] = sum(np.min(cdist(vec_ls_np, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / vec_ls_np.shape[0]
    mapping2[k] = kmeanModel.inertia_
      

for key, val in mapping1.items():
    print(f'{key} : {val}')
print()
    
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


for key, val in mapping2.items():
    print(f'{key} : {val}')
print() 

plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

#HC Clustering

col_ls=sites_further_analysis["site"].values.tolist()
sites_further_analysis_df=pd.DataFrame.from_dict(id_vec_dict, orient='index', columns=col_ls)


hc_min = hierarchy.linkage(sites_further_analysis_df,'single')
hierarchy.dendrogram(hc_min)
plt.title('HC dendrogram (MIN)')
plt.show()

hc_max = hierarchy.linkage(sites_further_analysis_df,'complete')
hierarchy.dendrogram(hc_max)
plt.title('HC dendrogram (MAX)')
plt.show()


hc_ward = hierarchy.linkage(sites_further_analysis_df,'ward')
hierarchy.dendrogram(hc_ward)
plt.axhline(y = 30, color = 'r', linestyle = '-')
plt.title('HC dendrogram (WARD)')
plt.show()

sites_further_analysis_df=sites_further_analysis_df.reset_index()
#Reindexing the df to include the record Id as a column
sites_further_analysis_df=sites_further_analysis_df.rename(columns={"index": "Record_Id"})
sites_further_analysis_df['hc_clusters'] = fcluster(hc_ward, 3, criterion='maxclust')
#Adding the cluster tag to each row

rec_hc_dict={}
for i in sites_further_analysis_df.index:
    rec_hc_dict[sites_further_analysis_df['Record_Id'][i]]=sites_further_analysis_df['hc_clusters'][i]
#Creating a dictionary for key=record Id, value=hc cluster

individuals['hc_clusters']=individuals['Record_Id'].map(rec_hc_dict)
#Adding the cluster tag to individuals df     
hc_col = individuals.pop('hc_clusters') 

individuals.insert(1, 'hc_clusters', hc_col) 
#Moving the cluster column to be at index 1 position

#Exploring the hc clusteing (ward) results: 
    
individuals['hc_clusters'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Clusters Size Distribution (ward method)")
 
rel1=[]
for i in individuals.index:
    if individuals['religion'][i]=='Christian':
        rel1.append('Christian')
    elif individuals['religion'][i]=='Jewish - Other':
        rel1.append('Jewish - Other')
    elif individuals['religion'][i]=='Jewish - Religious':
        rel1.append('Jewish - Religious')
    else:
        rel1.append('other')

def count_values_cat(plt_name):
    # iterate through axes
    for ax in plt_name.axes.ravel():       
        # add annotations
        for c in ax.containers:
            labels = [f'{(v.get_height())}' for v in c]
            ax.bar_label(c, labels=labels, label_type='edge')
        ax.margins(y=0.2)

individuals['religion_1']=rel1
order_ls_rel = ['Christian','Jewish - Religious','Jewish - Other','other']
ax1=sns.catplot(data=individuals,x='religion_1',kind='count',col='hc_clusters',order=order_ls_rel)
ax1.set_xticklabels(rotation=45)
ax1.fig.suptitle("Religion Distribution by Clusters",y=1.025)
count_values_cat(ax1)
plt.show()

n = 5
top5countries=individuals['country'].value_counts()[:n].index.tolist()
df1=individuals[['Record_Id','hc_clusters','country']]
df1=df1[df1['country'].isin(top5countries)]

ax2=sns.catplot(data=df1,x='country',kind='count',col='hc_clusters',order=df1.country.value_counts().index)
ax2.set_xticklabels(rotation=45)
ax2.fig.suptitle("Country Distribution by Clusters",y=1.025)
count_values_cat(ax2)
plt.show()

ax3=sns.catplot(data=individuals,x='country_category',kind='count',col='hc_clusters')
ax3.set_xticklabels(rotation=45)
ax3.fig.suptitle("Country Category Distribution by Clusters",y=1.025)
count_values_cat(ax3)
plt.show()

ax4=sns.catplot(data=individuals,x='first_visit',kind='count',col='hc_clusters')
ax4.fig.suptitle("First Visit- Y/N by Clusters",y=1.025)
count_values_cat(ax4)
plt.show()

individuals['short/long_visit'] = individuals['number_of_nights']
individuals['short/long_visit'].loc[individuals['number_of_nights']<=3] = 'short'
individuals['short/long_visit'].loc[individuals['number_of_nights']>3] = 'long'

ax5=sns.catplot(data=individuals,x='short/long_visit',kind='count',col='hc_clusters')
ax5.fig.suptitle("short/Long Visit by Clusters ",y=1.025)
count_values_cat(ax5)
plt.show()

individuals['group_bins']=individuals['group_size']
individuals['group_bins'].loc[individuals['group_size']==1] = 'sole tourist'
individuals['group_bins'].loc[individuals['group_size']==2] = 'couples'
individuals['group_bins'].loc[individuals['group_size']>2] = '3+'

ax6=sns.catplot(data=individuals,x='group_bins',kind='count',col='hc_clusters')
ax6.fig.suptitle("Group Size Distribution by Clusters",y=1.025)
count_values_cat(ax6)
plt.show()

individuals['sole_tourist'] = individuals['group_size']
individuals['sole_tourist'].loc[individuals['group_size']==1] = 'yes'
individuals['sole_tourist'].loc[individuals['group_size']>1] = 'no'

ax9=sns.catplot(data=individuals,x='sole_tourist',kind='count',col='hc_clusters')
ax9.fig.suptitle("sole tourist-y/n by Clusters",y=1.025)
count_values_cat(ax9)
plt.show()

rec_age_df=individuals.iloc[:,np.r_[0,19:33]]
rec_age_df=rec_age_df.rename(columns={"male_age_0_12": "1","male_age_13_18": "2","male_age_19_24": "3","male_age_25_39": "4","male_age_40_59": "5","male_age_60_74": "6","male_age_75plus": "7",'female_age_0_12':'a','female_age_13_18':'b','female_age_19_24':'c','female_age_25_39':'d','female_age_40_59':'e','female_age_60_74':'f','female_age_75plus':'g'})
age_dict={1:"0_12",2:"13_18",3:"19_59",4:"60_74",5:"75+"}
tourist_age_young={}
tourist_age_old={}

for i in rec_age_df.index:
    if rec_age_df['1'][i]==1 or rec_age_df['a'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[1]
        continue
    elif rec_age_df['7'][i]==1 or rec_age_df['g'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[5]
        continue
    elif rec_age_df['2'][i]==1 or rec_age_df['b'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[2]
        continue
    elif rec_age_df['6'][i]==1 or rec_age_df['f'][i]==1:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[4]
        continue
    else:
        tourist_age_young[individuals['Record_Id'][i]]=age_dict[3]

for i in rec_age_df.index:

    if rec_age_df['7'][i]==1 or rec_age_df['g'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[5]
        continue
    elif rec_age_df['1'][i]==1 or rec_age_df['a'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[1]
        continue
    elif rec_age_df['6'][i]==1 or rec_age_df['f'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[4]
        continue
    elif rec_age_df['2'][i]==1 or rec_age_df['b'][i]==1:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[2]
        continue
    else:
        tourist_age_old[individuals['Record_Id'][i]]=age_dict[3]
        
#old=75+ (oldest) / 60-74 (second)
#young=0_12 (youngest) / 13_18 (second)
#Creating an age tagging dictionary for each tourist that emplify the nost "extreme" individual (age wise), within a record.
#tourist_age_young gives more focus to the younger individual (if exist) while tourist_age_old gives more focus to the older individual (if exist).

individuals['age_tag_y']=individuals['Record_Id'].map(tourist_age_young)
individuals['age_tag_o']=individuals['Record_Id'].map(tourist_age_old)
#Adding the age tagging to individuals dataset

age_tag_y = individuals.pop('age_tag_y') 
individuals.insert(2, 'age_tag_y', age_tag_y) 
age_tag_o= individuals.pop('age_tag_o') 
individuals.insert(3, 'age_tag_o', age_tag_o) 
#Maping the age tagging columns to be at 2nd and 3rd index

order_list = ['0_12', '13_18', '19_59','60_74','75+']

ax7=sns.catplot(data=individuals,x='age_tag_y',kind='count',col='hc_clusters',order=order_list)
ax7.fig.suptitle("Age Distribution by Clusters (emphasis = younger) ",y=1.025)
count_values_cat(ax7)
plt.show

ax8=sns.catplot(data=individuals,x='age_tag_o',kind='count',col='hc_clusters',order=order_list)
ax8.fig.suptitle("Age Distribution by Clusters (emphasis = older)",y=1.025)
count_values_cat(ax8)
plt.show()

#####anomaly detection model########

#create df for anomality detection
removed_site_list=[]
general_site_list=individuals.iloc[:,97:-2].columns.values.tolist()#all sites list
finished_site_list=sites_further_analysis['site'].values.tolist()#filterd site list

for site in general_site_list:
    if site not in finished_site_list:
        removed_site_list.append(site)
site_df_anomalitiy=individuals.iloc[:,97:-2]
for site in site_df_anomalitiy.columns.values:
       if site in removed_site_list:
           site_df_anomalitiy=site_df_anomalitiy.drop([site], axis=1)# filted site df

def LofAnomalityDetecetion(df):
   from sklearn.neighbors import LocalOutlierFactor

   ground_truth = np.ones(len(df), dtype=int)

   clf = LocalOutlierFactor(n_neighbors=3, contamination=0.1)####In the results more neiboghrs less inliners
   y_pred = clf.fit_predict(df)  #1- inliner, -1 outlier
   n_errors = (y_pred).sum()
   X_scores = clf.negative_outlier_factor_  #The higher, the more normal. Inliers tend to have a LOF score close to 1 (negative_outlier_factor_ close to -1), 
                                           #while outliers tend to have a larger LOF score.
                                          #The local outlier factor (LOF) of a sample captures its supposed 
                                          #‘degree of abnormality’. It is the average of the ratio of the local 
   inlier_list=[]
   outlier_list=[]                                          #reachability density of a sample and those of its k-nearest neighbors.
   for num in y_pred:
       if num==1:
           inlier_list.append(num)
       else :
            outlier_list.append(num)
   print("sum of  LOF inliners  are " +str(len(inlier_list)))
   print("sum of  LOF outliers are "+ str(len(outlier_list)))
   
   return y_pred

def IsolationForest(df):
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(n_estimators=5, warm_start=True)
    clf.fit(df)  # fit 100 trees  
    clf.set_params(n_estimators=10)  # add 150 more trees  
    clf.fit(df)
    y_pred_isolationforest = clf.fit_predict(df)# fit the added trees 
    
    inlier_list=[]
    outlier_list=[]                                          #reachability density of a sample and those of its k-nearest neighbors.
    for num in  y_pred_isolationforest:
        if num==1:
            inlier_list.append(num)
        else:
             outlier_list.append(num)
    print("sum of IF inliners are " +str(len(inlier_list)))
    print("sum of IF outliers are "+ str(len(outlier_list)))
       
    return  y_pred_isolationforest

def SVMDetecetion(df):
  from sklearn.svm import OneClassSVM
#nu as a hyperparameter which is used to define what portion
# of data should be classified as outliers. nu = 0.03 means that the 
#algorithm will designate 3% data as outliers.

## gamma is used to set the kernel function for forming the hypersphere to learn and
# differnciate samples and the hyperparameter nu is tuned to approximate the ratio
# of outliers

  model = OneClassSVM( gamma = 'auto', nu = 0.05).fit(df)
  y_pred = model.predict(df)
                                    
  inlier_list=[]
  outlier_list=[]                                          #reachability density of a sample and those of its k-nearest neighbors.
  for num in y_pred:
       if num==1:
           inlier_list.append(num)
       else :
            outlier_list.append(num)
  print("sum of  SVM inliners  are " +str(len(inlier_list)))
  print("sum of  SVM outliers are "+ str(len(outlier_list)))
   
  return y_pred

def convert_to_percentages(df):
    df_percentages = pd.DataFrame()

    for column in df.columns:
        if df[column].dtype == 'object':
            column_total = df[column].count()
            percentages = df[column].value_counts() / column_total * 100
            df_percentages[column] = percentages.round(2)
        else:
            df_percentages[column] = df[column]

    df_percentages=df_percentages.reset_index()
    
    return df_percentages

def create_grouped_bar_chart(df, x_column, y_columns, legend_labels, title):

    x = df[x_column]  
    num_bars = len(y_columns)
    bar_width = 0.8 / num_bars

    fig, ax = plt.subplots()

    for i in range(num_bars):
        y = df[y_columns[i]]
        
        x_pos = [j + i * bar_width for j in range(len(x))] 
        ax.bar(x_pos, y, width=bar_width, label=legend_labels[i])
        
        
        for x_, y_ in zip(x_pos, y):
            rounded_y = floor(y_)
            ax.text(x_, y_ + 1, f'{rounded_y}%', ha='center', va='bottom')
            
    ax.set_xlabel(x_column)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticks([j + (num_bars - 1) * bar_width / 2 for j in range(len(x))])
    ax.set_xticklabels(x, rotation=90)
    ax.legend()
    
    plt.show()
#__________________________________________________________________________--

LOF=LofAnomalityDetecetion( site_df_anomalitiy)    

#-------------------------------------------------------------------------------------
Isolatioin_Forest=IsolationForest(site_df_anomalitiy)
#_________________________________________________________________________________________-

SVM=SVMDetecetion(site_df_anomalitiy)
#####Explore anomalities######
###LOF###
LOF = pd.DataFrame(LOF, columns = ['anomalities'])

site_df_anomalitiy.index = LOF.index

site_df_anomalitiy_LOF=pd.concat([site_df_anomalitiy, LOF], axis=1)

site_df_LOF_inliers = site_df_anomalitiy_LOF[site_df_anomalitiy_LOF['anomalities'] == 1]

site_df_LOF_outiers = site_df_anomalitiy_LOF[site_df_anomalitiy_LOF['anomalities'] == -1]

site_df_LOF_outiers = site_df_anomalitiy_LOF.iloc[:,0:-1]

site_df_LOF_inliers = site_df_LOF_inliers.iloc[:,0:-1]

####SVM####
svm= pd.DataFrame(SVM, columns = ['anomalities'])

site_df_anomalitiy.index = svm.index

site_df_anomalitiy_svm=pd.concat([site_df_anomalitiy, svm], axis=1)

site_df_svm_inliers = site_df_anomalitiy_svm[site_df_anomalitiy_svm['anomalities'] == 1]

site_df_svm_outiers = site_df_anomalitiy_svm[site_df_anomalitiy_svm['anomalities'] == -1]

site_df_svm_outiers = site_df_anomalitiy_svm.iloc[:,0:-1]

site_df_svm_inliers = site_df_svm_inliers.iloc[:,0:-1]

##Frequent sites outliers###
###checking asociation ruels
#########LOF######
print("-----------------------------------------------------------------------------------------------")

min_support_LOF = 0.1
min_confidence_LOF  = 0.6 
frequency_itemset_outliers=fpgrowth_asocsiation_rules(site_df_LOF_outiers, min_support_LOF, min_confidence_LOF, lift=True)
frequency_itemset_outliers=fpgrowth_asocsiation_rules(site_df_LOF_inliers, min_support_LOF, min_confidence_LOF, lift=True)

####top 10 sites outliers###
sumsitesoutliers=site_df_LOF_outiers.sum().sort_values(ascending=False)
print(sumsitesoutliers.head(10))

sumperpersonoutliers=site_df_LOF_outiers.sum(axis='columns'). mean()
print("average site per person outlier:"+ str(sumperpersonoutliers))

####top 10 sites inliers###
site_df_LOF_inliers_sum=site_df_LOF_inliers.sum().sort_values(ascending=False)
print(site_df_LOF_inliers_sum.head(10))

site_df_LOF_inliers_mean=site_df_LOF_inliers.sum(axis='columns'). mean()
print("average site per person inlier:"+ str(site_df_LOF_inliers_mean))

##################SVM#############

min_support_svm = 0.1
min_confidence_svm  = 0.6 
frequency_itemset_outliers=fpgrowth_asocsiation_rules(site_df_svm_outiers, min_support_svm, min_confidence_svm, lift=True)
frequency_itemset_outliers=fpgrowth_asocsiation_rules(site_df_svm_inliers, min_support_svm, min_confidence_svm, lift=True)

####top 10 sites outliers###
sumsitesoutliers=site_df_svm_outiers.sum().sort_values(ascending=False)
print(sumsitesoutliers.head(10))

sumperpersonoutliers=site_df_svm_outiers.sum(axis='columns'). mean()
print("average site per person outlier:"+ str(sumperpersonoutliers))

####top 10 sites inliers###
site_df_svm_inliers_sum=site_df_svm_inliers.sum().sort_values(ascending=False)
print(site_df_svm_inliers_sum.head(10))

site_df_svm_inliers_mean=site_df_svm_inliers.sum(axis='columns'). mean()
print("average site per person inlier:"+ str(site_df_svm_inliers_mean))

########looking at all of the data with SVM algorithm##### 
site_df_anomalitiy_svm_full=pd.concat([individuals.iloc[:,1:97],site_df_anomalitiy, svm], axis=1)

site_df_svm_inliers_full = site_df_anomalitiy_svm_full[site_df_anomalitiy_svm_full['anomalities'] == 1]

site_df_svm_outiers_full = site_df_anomalitiy_svm_full[site_df_anomalitiy_svm_full['anomalities'] == -1]

site_df_svm_outiers_full = site_df_svm_outiers_full.iloc[:,0:-1]

site_df_svm_inliers_full = site_df_svm_inliers_full.iloc[:,0:-1]

####bar plot of religion count for inlier and outliers
outlier_religon=site_df_svm_outiers_full['religion']
outlier_religon=outlier_religon.rename( 'outlier_religon')
inlier_religon=site_df_svm_inliers_full['religion']
inlier_religon= inlier_religon.rename('inlier_religon')
anomality_religion=pd.concat([outlier_religon,inlier_religon], axis=1)
outlier_religon.reset_index(drop=True, inplace=True)
inlier_religon.reset_index(drop=True, inplace=True)

df_percentages_religion = convert_to_percentages(anomality_religion)

create_grouped_bar_chart(df_percentages_religion, 'index', ['outlier_religon', 'inlier_religon'], ['outlier_religon',  'inlier_religon'], "religion precentage comparison svm")

####bar plot of country count for inlier and outliers
outlier_country=site_df_svm_outiers_full['country']
outlier_country=outlier_country.rename( 'outlier')
inlier_country=site_df_svm_inliers_full['country']
inlier_country= inlier_country.rename('inlier')
anomality_country=pd.concat([ outlier_country,inlier_country], axis=1)
top5_countries = anomality_country['outlier'].value_counts().nlargest(5).index
outlier_country = outlier_country[outlier_country.isin(top5_countries)] 
inlier_country = inlier_country[inlier_country.isin(top5_countries)]
anomality_country = pd.concat([outlier_country, inlier_country], axis=1)
outlier_country.reset_index(drop=True, inplace=True)
inlier_country.reset_index(drop=True, inplace=True)

df_percentages_country = convert_to_percentages(anomality_country)
create_grouped_bar_chart(df_percentages_country, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "country precentage comparison svm")

####bar plot of group size precentage for inlier and outliers
outlier_groupsize=site_df_svm_outiers_full['group_size']
outlier_groupsize=outlier_groupsize.rename( 'outlier')
inlier_groupsize=site_df_svm_inliers_full['group_size']
inlier_groupsize= inlier_groupsize.rename('inlier')

anomality_groupsize=pd.concat([ outlier_groupsize,inlier_groupsize], axis=1)
anomality_groupsize.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_groupsize = anomality_groupsize.astype('object')
df_percentages_groupsize = convert_to_percentages(anomality_groupsize)
create_grouped_bar_chart(df_percentages_groupsize, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "groupsize precentage comparison svm")

anomality_groupsize = anomality_groupsize.applymap(lambda x: 0 if pd.notna(x) and x != 1 else x)
df_counts = pd.DataFrame(index=[1, 0])
for column in anomality_groupsize:
    value_counts = anomality_groupsize[column].value_counts()
    df_counts[column] = value_counts.reindex([1, 0])
df_counts= df_counts.reset_index()

for column in df_counts.columns[1:]:
    total_count = df_counts[column].sum()
    df_counts[column] = df_counts[column] / total_count * 100

create_grouped_bar_chart(df_counts, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "solo vs group comparison svm")

####bar plot of first or returnes visit precentage for inlier and outliers
outlier_firstvisit=site_df_svm_outiers_full['first_visit']
outlier_firstvisit=outlier_firstvisit.rename( 'outlier')
inlier_firstvisit=site_df_svm_inliers_full['first_visit']
inlier_firstvisit= inlier_firstvisit.rename('inlier')

anomality_firstvisit=pd.concat([ outlier_firstvisit,inlier_firstvisit], axis=1)
anomality_firstvisit.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_firstvisit = anomality_firstvisit.astype('object')
df_percentages_firstvisit = convert_to_percentages(anomality_firstvisit)
create_grouped_bar_chart(df_percentages_firstvisit, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "firstvisit precentage comparison svm")

####bar plot of length_of_visit precentage for inlier and outliers
outlier_length_of_visit=site_df_svm_outiers_full['number_of_nights']
outlier_length_of_visit=outlier_length_of_visit.rename( 'outlier')
inlier_length_of_visit=site_df_svm_inliers_full['number_of_nights']
inlier_length_of_visit= inlier_length_of_visit.rename('inlier')

anomality_length_of_visit=pd.concat([ outlier_length_of_visit,inlier_length_of_visit], axis=1)
anomality_length_of_visit.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_length_of_visit = anomality_length_of_visit.astype('object')
df_percentages_length_of_visit = convert_to_percentages(anomality_length_of_visit)
create_grouped_bar_chart(df_percentages_length_of_visit, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "length_of_visit precentage comparison svm")

#####bar plot by gender####
####man
outlier_Number_of_men=site_df_svm_outiers_full['men_in_group']
outlier_Number_of_men=outlier_Number_of_men.rename( 'outlier')
inlier_Number_of_men=site_df_svm_inliers_full['men_in_group']
inlier_Number_of_men= inlier_Number_of_men.rename('inlier')

anomality_Number_of_men=pd.concat([ outlier_Number_of_men,inlier_Number_of_men], axis=1)
anomality_Number_of_men.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_Number_of_men = anomality_Number_of_men.astype('object')
df_percentages_Number_of_men = convert_to_percentages(anomality_Number_of_men)
create_grouped_bar_chart(df_percentages_Number_of_men, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "Number_of_men precentage comparison svm")

####women
outlier_Number_of_women=site_df_svm_outiers_full['women_in_group']
outlier_Number_of_women=outlier_Number_of_women.rename( 'outlier')
inlier_Number_of_women=site_df_svm_inliers_full['women_in_group']
inlier_Number_of_women= inlier_Number_of_women.rename('inlier')

anomality_Number_of_women=pd.concat([ outlier_Number_of_women,inlier_Number_of_women], axis=1)
anomality_Number_of_women.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_Number_of_women = anomality_Number_of_women.astype('object')
df_percentages_Number_of_women = convert_to_percentages(anomality_Number_of_women)
create_grouped_bar_chart(df_percentages_Number_of_women, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "Number_of_women precentage comparison svm")

####bar plot of age group precentage for inlier and outliers
####old
outlier_age_tag_o=site_df_svm_outiers_full['age_tag_o']
outlier_age_tag_o=outlier_age_tag_o.rename( 'outlier')
inlier_age_tag_o=site_df_svm_inliers_full['age_tag_o']
inlier_age_tag_o= inlier_age_tag_o.rename('inlier')

anomality_age_tag_o=pd.concat([ outlier_age_tag_o,inlier_age_tag_o], axis=1)
anomality_age_tag_o.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_age_tag_o = anomality_age_tag_o.astype('object')
df_percentages_age_tag_o = convert_to_percentages(anomality_age_tag_o)
create_grouped_bar_chart(df_percentages_age_tag_o, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "age_tag_o precentage comparison svm")

#######young

outlier_age_tag_y=site_df_svm_outiers_full['age_tag_y']
outlier_age_tag_y=outlier_age_tag_y.rename( 'outlier')
inlier_age_tag_y=site_df_svm_inliers_full['age_tag_y']
inlier_age_tag_y= inlier_age_tag_y.rename('inlier')

anomality_age_tag_y=pd.concat([ outlier_age_tag_y,inlier_age_tag_y], axis=1)
anomality_age_tag_y.fillna("nan")#print(anomality_groupsize.dtypes)
 
anomality_age_tag_y = anomality_age_tag_y.astype('object')
df_percentages_age_tag_y = convert_to_percentages(anomality_age_tag_y)
create_grouped_bar_chart(df_percentages_age_tag_y, 'index', ['outlier', 'inlier'],  ['outlier', 'inlier'], "age_tag_y precentage comparison svm")



