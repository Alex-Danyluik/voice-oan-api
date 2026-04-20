"""
Regression cases for single-turn voice answers that exceeded 100 words.

These cases are the latest 50 over-100-word assistant turns found in
conversations.md.

Run the live model-backed check with:
  VOICE_VERBOSE_ANSWER_INTEGRATION=1 pytest tests/verbose_answer_test.py -q
"""
from __future__ import annotations

import asyncio
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


MAX_SINGLE_TURN_WORDS = 200

VERBOSE_ANSWER_CASES = [
    {
        "conversation_id": "1774831",
        "assistant_turn_index": 11,
        "user_turn_index": 10,
        "started_at": "13/04/2026, 12:22:40 PM",
        "query": "પહેલા વેતરના વલા પડા પહેલા વેતરનાથ છે હજી હટમાં જ નથી આવ્યા અખંડ ખડાયા છે વયાણેલા જ નથી કેવા",
        "observed_answer_words": 388,
        "observed_answer_chars": 2046,
        "observed_question_marks": 5,
    },
    {
        "conversation_id": "1775831",
        "assistant_turn_index": 5,
        "user_turn_index": 4,
        "started_at": "13/04/2026, 01:01:46 PM",
        "query": "અમારી ગાયને પચતું નથી",
        "observed_answer_words": 574,
        "observed_answer_chars": 3009,
        "observed_question_marks": 16,
    },
    {
        "conversation_id": "1776110",
        "assistant_turn_index": 3,
        "user_turn_index": 2,
        "started_at": "13/04/2026, 01:14:16 PM",
        "query": "સરનાદન મારી વાછળી છે એને વરદાનદાન ને મિલર પાવડર ખોરવાથી શરીરમાં કંઈ ફેરફાર થઈ શકે એનો ઉપચાર જણાવજો",
        "observed_answer_words": 514,
        "observed_answer_chars": 2790,
        "observed_question_marks": 12,
    },
    {
        "conversation_id": "1784382",
        "assistant_turn_index": 5,
        "user_turn_index": 4,
        "started_at": "13/04/2026, 07:33:39 PM",
        "query": (
            "બો મેડમ મારી બ્યાસ પાંચ દિવસ થયા તેવાય એનું બચ્ચું નાનું છે તે પોધરો કરે છે તે લોહી જેવું આવે છે "
            "કઈ પ્રોબ્લમ થાયો હલોાલને બોલ બોલ હા મેડમ બ તો પાંચ દિવસ થયા છે એનું બચ્ચું નાનું છે તે પોધરો કરે છે "
            "ને એમાં લોહી જેવું આવે છે તો કોઈ પ્રોબ્લેમ થાય"
        ),
        "observed_answer_words": 420,
        "observed_answer_chars": 2271,
        "observed_question_marks": 6,
    },
    {
        "conversation_id": "1784382",
        "assistant_turn_index": 7,
        "user_turn_index": 6,
        "started_at": "13/04/2026, 07:33:39 PM",
        "query": "ત્યારને બચ્ચા ને મેડમ લોહી આવે છે ત્યારે સંડાસ કરે ને તારે પેલી વાર આવ તો કોઈ પ્રોબ્લેમ થઈ શકે બચ્ચાને",
        "observed_answer_words": 243,
        "observed_answer_chars": 1317,
        "observed_question_marks": 8,
    },
    {
        "conversation_id": "1784382",
        "assistant_turn_index": 9,
        "user_turn_index": 8,
        "started_at": "13/04/2026, 07:33:39 PM",
        "query": "આ પ્યાલી જ વાર થયું છે બચ્ચાને જાાડા ને પાતળય નથી ત જાડા નથી બરમાં છે",
        "observed_answer_words": 354,
        "observed_answer_chars": 1873,
        "observed_question_marks": 8,
    },
    {
        "conversation_id": "1784382",
        "assistant_turn_index": 13,
        "user_turn_index": 12,
        "started_at": "13/04/2026, 07:33:39 PM",
        "query": "બચ્ચાને બચ્ચાનોો જન્મ પાંચ દિવસ થયા છે",
        "observed_answer_words": 281,
        "observed_answer_chars": 1482,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1784414",
        "assistant_turn_index": 7,
        "user_turn_index": 6,
        "started_at": "13/04/2026, 09:49:09 PM",
        "query": "આ એક ગિરી ગાય છે એ રેતી",
        "observed_answer_words": 192,
        "observed_answer_chars": 980,
        "observed_question_marks": 4,
    },
    {
        "conversation_id": "1784414",
        "assistant_turn_index": 9,
        "user_turn_index": 8,
        "started_at": "13/04/2026, 09:49:09 PM",
        "query": "હલો ગાય રેતી નથી ઉભતી ગરમીમાં નથી આવતી ને ઘા પણ નથી થતી એમ કહું ને બાકી ત ચિતત ડોક્ટર ન કોણ તો એે વાો ડોક્ટર આવતો નથી અમે",
        "observed_answer_words": 641,
        "observed_answer_chars": 3358,
        "observed_question_marks": 10,
    },
    {
        "conversation_id": "1835377",
        "assistant_turn_index": 14,
        "user_turn_index": 12,
        "started_at": "18/04/2026, 07:31:31 AM",
        "query": (
            "આ લાઈન પર બની રહી ધ પર્સન યુ આર સ્પીકિંગ વિથ હેઝ પુટ યર કોલ ઓન હોલ્ડ પ્લીઝ સ્ટે ઓન ધ લાઈન "
            "તમે જે વ્યક્તિને કોલ કર્યો છે તેને તમારો કોલ હો"
        ),
        "observed_answer_words": 172,
        "observed_answer_chars": 989,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835380",
        "assistant_turn_index": 11,
        "user_turn_index": 9,
        "started_at": "18/04/2026, 07:37:03 AM",
        "query": "મારી ગાયને બીજદાન કરાવવાનું છે",
        "observed_answer_words": 117,
        "observed_answer_chars": 653,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835386",
        "assistant_turn_index": 10,
        "user_turn_index": 8,
        "started_at": "18/04/2026, 07:44:17 AM",
        "query": "સલાબેન ગાયને ચૂનાનું પાણી પીવડાવીએ તો ફાયદો થાય કે નુકસાન",
        "observed_answer_words": 176,
        "observed_answer_chars": 1006,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835386",
        "assistant_turn_index": 17,
        "user_turn_index": 15,
        "started_at": "18/04/2026, 07:44:17 AM",
        "query": "સરલાબેન ગુજરાતી માં બોલો",
        "observed_answer_words": 174,
        "observed_answer_chars": 963,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1835383",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 07:39:43 AM",
        "query": "હા મારા પશુને રસી મુકાવી છે તો શું કરવું",
        "observed_answer_words": 307,
        "observed_answer_chars": 1895,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835383",
        "assistant_turn_index": 17,
        "user_turn_index": 15,
        "started_at": "18/04/2026, 07:39:43 AM",
        "query": "મારી જોડે ભસ છે એને એપ્રિલ મહિનામાં કઈ રસી મુકાવું",
        "observed_answer_words": 320,
        "observed_answer_chars": 1782,
        "observed_question_marks": 4,
    },
    {
        "conversation_id": "1835383",
        "assistant_turn_index": 22,
        "user_turn_index": 20,
        "started_at": "18/04/2026, 07:39:43 AM",
        "query": "ઉપરની બધી જ ઇન્ફોર્મેશન ગુજરાતીમાં આપો",
        "observed_answer_words": 423,
        "observed_answer_chars": 2238,
        "observed_question_marks": 5,
    },
    {
        "conversation_id": "1835388",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 07:47:12 AM",
        "query": "મારે ઉચ્ચ ઓલાદની વાછરડી મેળવવી હોય તો શું કરવું જોઈએ",
        "observed_answer_words": 685,
        "observed_answer_chars": 4187,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835398",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 09:39:57 AM",
        "query": "મારો ગાયને દૂધ વધારવા શું કરવું જોઈએ",
        "observed_answer_words": 670,
        "observed_answer_chars": 3980,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835395",
        "assistant_turn_index": 12,
        "user_turn_index": 10,
        "started_at": "18/04/2026, 09:38:48 AM",
        "query": "ભાઈ પાચું વંધર્ય નિવારણ માટે શું કરવું",
        "observed_answer_words": 350,
        "observed_answer_chars": 2139,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1835400",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 09:41:11 AM",
        "query": "વાછળ નો ઉછેર કેવી રીતે કરવો",
        "observed_answer_words": 779,
        "observed_answer_chars": 4464,
        "observed_question_marks": 1,
    },
    {
        "conversation_id": "1835516",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 10:04:08 AM",
        "query": "અ સરળાબેન મારે નવ નવચાત જન્મેલ વાછરડાના શીંગડા ડામવા છે તો શું કરવાનું ક્યાં કેટલા દિવસમાં ડામી શકાય",
        "observed_answer_words": 718,
        "observed_answer_chars": 4138,
        "observed_question_marks": 1,
    },
    {
        "conversation_id": "1835516",
        "assistant_turn_index": 9,
        "user_turn_index": 7,
        "started_at": "18/04/2026, 10:04:08 AM",
        "query": "હા સરલાબેન મારે અ મારી ગાયને બગાઈ બો લાગેલી છે તો એ કેવી રીતના દૂર કરી શકાય કોઈ ઘરગથ્થુ ઉપચાર બતાવશો",
        "observed_answer_words": 625,
        "observed_answer_chars": 3568,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1836488",
        "assistant_turn_index": 13,
        "user_turn_index": 11,
        "started_at": "18/04/2026, 10:44:47 AM",
        "query": "અત્યારે ઉનાળો છે અને ગરમી બહુ છે તો મારેજે ગાય ભેસ માટે રહેવાની વ્યવસ્થા કેવી કરવી જોઈએ જેનાથી એ થોડી ઠંડક રહે",
        "observed_answer_words": 477,
        "observed_answer_chars": 2635,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1836488",
        "assistant_turn_index": 16,
        "user_turn_index": 14,
        "started_at": "18/04/2026, 10:44:47 AM",
        "query": "મારે મારે ગાયના દૂધમાં છે ને ખોદા જેવું આવે છે તો શું હોઈ શકે થોડું જણાવશો",
        "observed_answer_words": 366,
        "observed_answer_chars": 2033,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1836488",
        "assistant_turn_index": 19,
        "user_turn_index": 17,
        "started_at": "18/04/2026, 10:44:47 AM",
        "query": (
            "અ આજકાલ મારી એક ગાય છે એ છે ને દૂધ આપવામાં બહુ નાટક કરે છે પગ વડે શું કેહેવાય લાતો મારે છે "
            "ને એવું તો મારે શું કરવું જોઈએ જેનાથી શાંતિથી દૂધ આપી શકે"
        ),
        "observed_answer_words": 537,
        "observed_answer_chars": 2880,
        "observed_question_marks": 7,
    },
    {
        "conversation_id": "1836488",
        "assistant_turn_index": 25,
        "user_turn_index": 23,
        "started_at": "18/04/2026, 10:44:47 AM",
        "query": "એક ગાય છે એના આંખમાંથી પાણી નીકળે છે તો શું હોય શકે",
        "observed_answer_words": 355,
        "observed_answer_chars": 1930,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1838031",
        "assistant_turn_index": 7,
        "user_turn_index": 5,
        "started_at": "18/04/2026, 11:49:56 AM",
        "query": "દુજ વધારવા શું કરવું",
        "observed_answer_words": 350,
        "observed_answer_chars": 1977,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1839575",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 12:58:59 PM",
        "query": "ઘઉનું ભૂસું કઈ સિઝનમાં આપવું વધુ જતાવળ છે",
        "observed_answer_words": 207,
        "observed_answer_chars": 1127,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1839665",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 01:02:17 PM",
        "query": "ઉનાળાની ઋતુ દરમિયાન ગરમીની સિઝનમાં પશુને કેવો સૂકો અને કેવો લીલો ચારો આપવો જોઈએ",
        "observed_answer_words": 380,
        "observed_answer_chars": 2092,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1839665",
        "assistant_turn_index": 10,
        "user_turn_index": 8,
        "started_at": "18/04/2026, 01:02:17 PM",
        "query": "ઉનાળાની ઋતુ દરમિયાન પશુને લશકા બાજરી ગરમ પડે કે કેમ",
        "observed_answer_words": 263,
        "observed_answer_chars": 1463,
        "observed_question_marks": 4,
    },
    {
        "conversation_id": "1839665",
        "assistant_turn_index": 13,
        "user_turn_index": 11,
        "started_at": "18/04/2026, 01:02:17 PM",
        "query": "ચણાનું ભૂસું કઈ સિગમમાં આપવું વધુ રીતા હોય છે",
        "observed_answer_words": 271,
        "observed_answer_chars": 1425,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1839665",
        "assistant_turn_index": 18,
        "user_turn_index": 16,
        "started_at": "18/04/2026, 01:02:17 PM",
        "query": "લસકા બાજરી છે ઘઉંનું ભૂસું છે અને ચણાનું ભૂસું છે ગાય અને ભેંસ બને છે",
        "observed_answer_words": 346,
        "observed_answer_chars": 1787,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1840795",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 01:47:56 PM",
        "query": "મારી બેસને પાંચ છ વખત ડોઝ મુકાયો કૃત્રિમ બદદાન કરાયું તો પાછી ફરે છે શું કરું",
        "observed_answer_words": 986,
        "observed_answer_chars": 5819,
        "observed_question_marks": 7,
    },
    {
        "conversation_id": "1840758",
        "assistant_turn_index": 10,
        "user_turn_index": 8,
        "started_at": "18/04/2026, 01:46:29 PM",
        "query": "અમારે લોન જોઈએ છેએ",
        "observed_answer_words": 113,
        "observed_answer_chars": 644,
        "observed_question_marks": 6,
    },
    {
        "conversation_id": "1840758",
        "assistant_turn_index": 13,
        "user_turn_index": 11,
        "started_at": "18/04/2026, 01:46:29 PM",
        "query": "ંગાઈ માટે પાંચ લાખની લોન ની જરૂર છે",
        "observed_answer_words": 506,
        "observed_answer_chars": 2909,
        "observed_question_marks": 8,
    },
    {
        "conversation_id": "1841289",
        "assistant_turn_index": 7,
        "user_turn_index": 5,
        "started_at": "18/04/2026, 02:08:13 PM",
        "query": "કડી સેટીમાં ડેરી લાવવી છે",
        "observed_answer_words": 717,
        "observed_answer_chars": 4185,
        "observed_question_marks": 5,
    },
    {
        "conversation_id": "1850005",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 08:04:19 PM",
        "query": "મિનરલ પાવડર કયો વાપરવા માટે",
        "observed_answer_words": 354,
        "observed_answer_chars": 2032,
        "observed_question_marks": 6,
    },
    {
        "conversation_id": "1850005",
        "assistant_turn_index": 9,
        "user_turn_index": 7,
        "started_at": "18/04/2026, 08:04:19 PM",
        "query": "દૂધ આપતી ગાય માટે કયો મિનરલ પાવડર વાપરોવો",
        "observed_answer_words": 255,
        "observed_answer_chars": 1456,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1850005",
        "assistant_turn_index": 15,
        "user_turn_index": 13,
        "started_at": "18/04/2026, 08:04:19 PM",
        "query": "પાંચ લીટર દૂધ આપે છે ગાય અને ઘાસચારામાં મકાઈનું સૂકું ઘાસ છે અને ઘઉં ન ભૂસું છે અને પંચામૃત દાણ",
        "observed_answer_words": 213,
        "observed_answer_chars": 1183,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1850014",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "18/04/2026, 10:12:01 PM",
        "query": "ગરમીથી પુ બચ ગરમીથી પશુ બચાવવા શું કરવું પડે",
        "observed_answer_words": 489,
        "observed_answer_chars": 2674,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1850020",
        "assistant_turn_index": 7,
        "user_turn_index": 5,
        "started_at": "18/04/2026, 10:36:30 PM",
        "query": "ભેસને ગરમી બેટ મુદ ઋતુમાં આવે કરવાની થાય સમય સમય",
        "observed_answer_words": 600,
        "observed_answer_chars": 3245,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1850044",
        "assistant_turn_index": 12,
        "user_turn_index": 10,
        "started_at": "19/04/2026, 09:22:58 AM",
        "query": "સલાબેન મને ટેગ નંબર ત્રણ ચાર જીરો એક સાત એક શૂન્ય ચાર બે જીરો બે સાત ની માહિતી આપો",
        "observed_answer_words": 111,
        "observed_answer_chars": 565,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1850044",
        "assistant_turn_index": 18,
        "user_turn_index": 16,
        "started_at": "19/04/2026, 09:22:58 AM",
        "query": "ત્રણ ચાર શૂન્ય એક સાત એક શૂન્ય ચાર બે શૂન્ય બે સાત ની માહિતી આપો",
        "observed_answer_words": 123,
        "observed_answer_chars": 631,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1850044",
        "assistant_turn_index": 21,
        "user_turn_index": 19,
        "started_at": "19/04/2026, 09:22:58 AM",
        "query": "સારલાબેન આા કાનની કડીનોો નંબર હતો તો તેની માહિતી આપો મને",
        "observed_answer_words": 196,
        "observed_answer_chars": 1040,
        "observed_question_marks": 1,
    },
    {
        "conversation_id": "1853149",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "19/04/2026, 12:11:00 PM",
        "query": "એ ફે હીટમાં નથી આવતી તો એને કઈ દવા અપાય નામ જણાવો ને",
        "observed_answer_words": 457,
        "observed_answer_chars": 2588,
        "observed_question_marks": 3,
    },
    {
        "conversation_id": "1853149",
        "assistant_turn_index": 17,
        "user_turn_index": 15,
        "started_at": "19/04/2026, 12:11:00 PM",
        "query": "એક કે વેતર વયાણી નથી પાડી છે હાડા ચાર વર્ષની ખરાાય થઈ ગઈ છે એને ગાભણી કરવા માટે કઈ દવા આપવી એટલા માટે પૂછ્યું છે",
        "observed_answer_words": 438,
        "observed_answer_chars": 2410,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1860076",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "19/04/2026, 10:35:55 PM",
        "query": "મારી ગાયનું દૂધ ઓછું છે તો કઈ રીતે વધારી શ ભાઈ",
        "observed_answer_words": 398,
        "observed_answer_chars": 2181,
        "observed_question_marks": 2,
    },
    {
        "conversation_id": "1860095",
        "assistant_turn_index": 4,
        "user_turn_index": 2,
        "started_at": "20/04/2026, 07:58:51 AM",
        "query": "ગાય થકી નથી",
        "observed_answer_words": 468,
        "observed_answer_chars": 2740,
        "observed_question_marks": 11,
    },
    {
        "conversation_id": "1860094",
        "assistant_turn_index": 7,
        "user_turn_index": 5,
        "started_at": "20/04/2026, 07:56:01 AM",
        "query": "ગાયને આંખમાં ચીપડા હોય એવું લાગે છે તો શું કરી શકાય",
        "observed_answer_words": 332,
        "observed_answer_chars": 1738,
        "observed_question_marks": 0,
    },
    {
        "conversation_id": "1860094",
        "assistant_turn_index": 10,
        "user_turn_index": 8,
        "started_at": "20/04/2026, 07:56:01 AM",
        "query": "ચલા બેન વાછડી જન્મી પછી મેલી પડતી નથી ગાય ર",
        "observed_answer_words": 447,
        "observed_answer_chars": 2440,
        "observed_question_marks": 1,
    },
]


def _case_id(case: dict) -> str:
    return f"{case['conversation_id']}-{case['assistant_turn_index']}"


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text.strip()))


async def _collect_live_voice_answer(case: dict, monkeypatch) -> str:
    from app.services import voice as voice_module

    history_store: dict[str, list] = {}

    async def _get_or_fetch_farmer_data(_mobile):
        return None

    async def _update_message_history(session_id, messages):
        history_store[session_id] = messages

    async def _send_nudge_message_raya(message, session_id, process_id=None):
        return None

    monkeypatch.setattr(voice_module, "normalize_phone_to_mobile", lambda user_id: None)
    monkeypatch.setattr(voice_module, "get_or_fetch_farmer_data", _get_or_fetch_farmer_data)
    monkeypatch.setattr(voice_module, "update_message_history", _update_message_history)
    monkeypatch.setattr(voice_module, "send_nudge_message_raya", _send_nudge_message_raya)
    monkeypatch.setattr(voice_module.settings, "nudge_timeout_seconds", 30.0, raising=False)

    chunks: list[str] = []
    case_id = _case_id(case)
    async for chunk in voice_module.stream_voice_message(
        query=case["query"],
        session_id=f"verbose-answer-{case_id}",
        source_lang="gu",
        target_lang="gu",
        user_id="anonymous",
        history=[],
        provider=None,
        process_id=f"verbose-answer-{case_id}",
        user_info={},
        owner=None,
        http_request=None,
    ):
        if isinstance(chunk, str):
            chunks.append(chunk)
    return "".join(chunks)


def test_fixture_has_latest_50_single_turn_over_100_word_answers():
    assert len(VERBOSE_ANSWER_CASES) == 50
    assert all(case["observed_answer_words"] > MAX_SINGLE_TURN_WORDS for case in VERBOSE_ANSWER_CASES)
    assert VERBOSE_ANSWER_CASES[-1]["started_at"].startswith("20/04/2026")


@pytest.mark.skipif(
    not _env_flag("VOICE_VERBOSE_ANSWER_INTEGRATION"),
    reason="Set VOICE_VERBOSE_ANSWER_INTEGRATION=1 to run live verbose-answer regressions",
)
@pytest.mark.parametrize("case", VERBOSE_ANSWER_CASES[:10], ids=_case_id)
def test_verbose_cases_run_through_voice_pipeline_under_100_words(case, monkeypatch):
    answer = asyncio.run(_collect_live_voice_answer(case, monkeypatch))
    answer_words = _word_count(answer)

    assert answer.strip(), f"{_case_id(case)} returned an empty answer"
    assert answer_words <= MAX_SINGLE_TURN_WORDS, (
        f"{_case_id(case)} old production answer had {case['observed_answer_words']} words; "
        f"live answer has {answer_words} words. Answer: {answer[:800]}"
    )
