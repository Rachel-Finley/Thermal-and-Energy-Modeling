import torch
from tqdm import tqdm
from tokenizers import Tokenizer
from torch.optim.lr_scheduler import LinearLR
import numpy as np
from transformers import BertConfig, AutoTokenizer, AutoModelForQuestionAnswering
import random


class BERT_QA():
    '''class for running QA model on the summaries'''

    def __init__(self):
        '''initialize models, and variables'''

        # initialize GPU
        self.device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

        # models
        self.tokenizer = AutoTokenizer.from_pretrained("csarron/bert-base-uncased-squad-v1")
        self.model = AutoModelForQuestionAnswering.from_pretrained("csarron/bert-base-uncased-squad-v1")

        # send model to GPU
        self.model.to(self.device)

        # encoded text data, model inputs, embeddings, text
        self.encoding = {}
        self.inputs = {}
        self.sentence_embedding = {}
        self.tokens = {}

        # index of QA model's answer
        self.start_scores = {}
        self.end_scores = {}
        self.start_index = {}
        self.end_index = {}
        self.outputs = {}

        # predicted answer's text
        self.init_answer = ""
        self.cleaned_answer = ""
        self.answer_list = []

        # preprocessing variables to save split text data
        self.split_total = []
        self.split_partial = []

    def get_split_tokens(self, context):
        '''splits text from wiki summaries to be right size for BERT model'''

        if len(context.split()) // 150 > 0:
            n = len(context.split()) // 150
        else:
            n = 1
        for w in range(n):
            if w == 0:
                self.split_partial = context.split()[:200]
                self.split_total.append(" ".join(self.split_partial))
            else:
                self.split_partial = context.split()[w * 150:w * 150 + 200]
                self.split_total.append(" ".join(self.split_partial))

    def pre_process(self, question, split_wiki_summaries):
        '''tokenizes and encodes question/context for BERT model'''

        self.encoding = self.tokenizer.encode_plus(text=question, text_pair=self.split_total[0])
        # token embeddings
        self.inputs = self.encoding['input_ids']
        # segment embeddings
        self.sentence_embedding = self.encoding['token_type_ids']
        # input tokens
        self.tokens = self.tokenizer.convert_ids_to_tokens(self.inputs)

    def predict(self):
        '''uses BERT model to predict the span of text with the answer'''

        # run QA model to index predicted answer
        for batch in tqdm(range(len(self.tokens))):  # Iterate over batches
            self.outputs = self.model(
                input_ids=torch.tensor([self.inputs]).to(self.device),  # Use the batch input_ids
                token_type_ids=torch.tensor([self.sentence_embedding]).to(self.device)  # Use the batch token_type_ids
            )
            self.start_index = torch.argmax(self.outputs.start_logits).to(self.device)
            self.end_index = torch.argmax(self.outputs.end_logits).to(self.device)

            # return predicted answer as string
            self.init_answer = ' '.join(self.tokens[self.start_index: self.end_index + 1])
            for word in self.init_answer.split():
                if word[0:2] == '##':
                    self.cleaned_answer += word[2:]
                else:
                    self.cleaned_answer += ' ' + word

        # add the answer to the list
        self.answer_list.append(self.init_answer)
        self.init_answer = ""
        self.split_total = []


def main():
    contexts = [
        '''The Treaty of Versailles was a peace treaty signed on 28 June 1919.
        As the most important treaty of World War I, it ended the state of war between Germany and most of the Allied Powers.
        It was signed in the Palace of Versailles, exactly five years after the assassination of Archduke Franz Ferdinand,
        which led to the war. The other Central Powers on the German side signed separate treaties.  The United States never
        ratified the Versailles treaty and made a separate peace treaty with Germany.  Although the armistice of 11 November
        1918 ended the actual fighting, it took six months of Allied negotiations at the Paris Peace Conference to conclude the
        peace treaty. Germany was not allowed to participate in the negotiations—it was forced to sign the final result.
        \nThe most critical and controversial provision in the treaty was: "The Allied and Associated Governments affirm and
        Germany accepts the responsibility of Germany and her allies for causing all the loss and damage to which the Allied
        and Associated Governments and their nationals have been subjected as a consequence of the war imposed upon them by
        the aggression of Germany and her allies." The other members of the Central Powers signed treaties containing similar
        articles. This article, Article 231, became known as the War Guilt clause. The treaty required Germany to disarm, make
        ample territorial concessions, and pay reparations to certain countries that had formed the Entente powers. In 1921 the
        total cost of these reparations was assessed at 132 billion gold marks (then $31.4 billion or £6.6 billion, roughly
        equivalent to US$442 billion or UK£284 billion in 2023).  Because of the way the deal was structured, the Allied Powers
        intended Germany would only ever pay a value of 50 billion marks.\nProminent economists such as John Maynard Keynes
        declared the treaty too harsh—a "Carthaginian peace"—and said the reparations were excessive and counter-productive.
        On the other hand, prominent Allied figures such as French Marshal Ferdinand Foch, criticized the treaty for treating
        Germany too leniently. This is still the subject of ongoing debate by historians and economists.The result of these
        competing and sometimes conflicting goals among the victors was a compromise that left no one satisfied. In particular,
        Germany was neither pacified nor conciliated, nor was it permanently weakened. The problems that arose from the treaty
        would lead to the Locarno Treaties, which improved relations between Germany and the other European powers. The
        reparation system was reorganized resulting in the Dawes Plan, the Young Plan, and the indefinite postponement of
        reparations at the Lausanne Conference of 1932. The treaty\'s terms against Germany resulted in economic collapse and
        bitter resentment which powered the rise of the Nazi Party, and eventually the outbreak of a second World War.
        \nAlthough it is often referred to as the "Versailles Conference", only the actual signing of the treaty took place at
        the historic palace. Most of the negotiations were in Paris, with the "Big Four" meetings taking place generally at the
        French Ministry of Foreign Affairs on the Quai d\'Orsay.', 1: 'Article 231, often known as the War Guilt Clause, was the
        opening article of the reparations section of the Treaty of Versailles, which ended the First World War between the
        German Empire and the Allied and Associated Powers. The article did not use the word "guilt" but it served as a legal
        basis to compel Germany to pay reparations for the war.\nArticle 231 was one of the most controversial points of the
        treaty. It specified: \n\n"The Allied and Associated Governments affirm and Germany accepts the responsibility of
        Germany and her allies for causing all the loss and damage to which the Allied and Associated Governments and their
        nationals have been subjected as a consequence of the war imposed upon them by the aggression of Germany and her
        allies."Germans viewed this clause as a national humiliation, forcing Germany to accept full responsibility for causing
        the war. German politicians were vocal in their opposition to the article in an attempt to generate international
        sympathy, while German historians worked to undermine the article with the objective of subverting the entire treaty.
        The Allied leaders were surprised at the German reaction; they saw the article only as a necessary legal basis to
        extract compensation from Germany. The article, with the signatory\'s name changed, was also included in the treaties
        signed by Germany\'s allies who did not view the clause with the same disdain as the Germans did. American diplomat
        John Foster Dulles—one of the two authors of the article—later regretted the wording used, believing it further
        aggravated the German people.\nThe historical consensus is that responsibility or guilt for the war was not attached
        to the article. Rather, the clause was a prerequisite to allow a legal basis to be laid out for the reparation payments
        that were to be made. Historians have also highlighted the unintended damage created by the clause, which caused anger
        and resentment amongst the German population.', 2: 'Following the ratification of article 231 of the Treaty of
        Versailles at the conclusion of World War I, the Central Powers were made to give war reparations to the Allied Powers.
        Each of the defeated powers was required to make payments in either cash or kind. Because of the financial situation in
        Austria, Hungary, and Turkey after the war, few to no reparations were paid and the requirements for reparations were
        cancelled. Bulgaria, having paid only a fraction of what was required, saw its reparation figure reduced and then
        cancelled. Historians have recognized the German requirement to pay reparations as the "chief battleground of the
        post-war era" and "the focus of the power struggle between France and Germany over whether the Versailles Treaty was
        to be enforced or revised."The Treaty of Versailles (signed in 1919) and the 1921 London Schedule of Payments required
        Germany to pay 132 billion gold marks (US$33 billion [all values are contemporary, unless otherwise stated]) in
        reparations to cover civilian damage caused during the war. This figure was divided into three categories of bonds:
        A, B, and C. Of these, Germany was required to pay towards \'A\' and \'B\' bonds totaling 50 billion marks
        (US$12.5 billion) unconditionally. The payment of the remaining \'C\' bonds was interest free and contingent on the
        Weimar Republic\'s ability to pay, as was to be assessed by an Allied committee.\nDue to the lack of reparation
        payments by Germany, France occupied the Ruhr in 1923 to enforce payments, causing an international crisis that
        resulted in the implementation of the Dawes Plan in 1924. This plan outlined a new payment method and raised
        international loans to help Germany to meet its reparation commitments. Despite this, by 1928 Germany called for a
        new payment plan, resulting in the Young Plan that established the German reparation requirements at 112 billion marks
        (US$26.3 billion) and created a schedule of payments that would see Germany complete payments by 1988. With the collapse
        of the German economy in 1931, reparations were suspended for a year and in 1932 during the Lausanne Conference they
        were cancelled altogether. Between 1919 and 1932, Germany paid less than 21 billion marks in reparations.\nThe German
        people saw reparations as a national humiliation; the German Government worked to undermine the validity of the Treaty
        of Versailles and the requirement to pay. British economist John Maynard Keynes called the treaty a Carthaginian peace
        that would economically destroy Germany. His arguments had a profound effect on historians, politicians, and the public
        at large. Despite Keynes\' arguments and those by later historians supporting or reinforcing Keynes\' views, the
        consensus of contemporary historians is that reparations were not as intolerable as the Germans or Keynes had suggested
        and were within Germany\'s capacity to pay had there been the political will to do so. Following the Second World War,
        West Germany took up payments. The 1953 London Agreement on German External Debts resulted in an agreement to pay 50
        percent of the remaining balance. The final payment was made on 3 October 2010, settling German loan debts in regard to
        reparations.', 3: 'The U.S.–German Peace Treaty was a peace treaty between the U.S. and the German governments. It was
        signed in Berlin on August 25, 1921 in the aftermath of World War I. The main reason for the conclusion of that treaty
        was the fact that the U.S. Senate did not consent to ratification of the multilateral peace treaty signed in Versailles,
        thus leading to a separate peace treaty. Ratifications were exchanged in Berlin on November 11, 1921, and the treaty
        became effective on the same day. The treaty was registered in League of Nations Treaty Series on August 12, 1922.''',
        '''The assassination of Archduke Franz Ferdinand is considered one of the key events that led to World War I. Archduke 
        Franz Ferdinand of Austria, heir presumptive to the Austro-Hungarian throne, and his wife, Sophie, Duchess of Hohenberg, 
        were assassinated on 28 June 1914 by Bosnian Serb student Gavrilo Princip. They were shot at close range while being 
        driven through Sarajevo, the provincial capital of Bosnia-Herzegovina, formally annexed by Austria-Hungary in 1908.
        Princip was part of a group of six Bosnian assassins together with Muhamed Mehmedbašić, Vaso Čubrilović, Nedeljko Čabrinović, 
        Cvjetko Popović and Trifko Grabež coordinated by Danilo Ilić; all but one were Bosnian Serbs and members of a student 
        revolutionary group that later became known as Young Bosnia. The political objective of the assassination was to free 
        Bosnia and Herzegovina of Austria-Hungarian rule and establish a common South Slav ("Yugoslav") state. The assassination 
        precipitated the July Crisis which led to Austria-Hungary declaring war on Serbia and the start of World War I.
        The assassination team was helped by the Black Hand, a Serbian secret nationalist group; support came from Dragutin 
        Dimitrijević, chief of the military intelligence section of the Serbian general staff, as well as from Major Vojislav 
        Tankosić and Rade Malobabić, a Serbian intelligence agent. Tankosić provided bombs and pistols to the assassins and trained 
        them in their use. The assassins were given access to the same clandestine network of safe-houses and agents that Malobabić 
        used for the infiltration of weapons and operatives into Austria-Hungary.
        The assassins and key members of the clandestine network were tried in Sarajevo in October 1914. In total twenty-five people 
        were indicted. All six assassins, except Mehmedbašić, were under twenty at the time of the assassination; while the group was 
        dominated by Bosnian Serbs, four of the indictees were Bosnian Croats, and all of them were Austro-Hungarian citizens, none from 
        Serbia. Princip was found guilty of murder and high treason; too young to be executed, he was sentenced to twenty years in jail, 
        while the four other attackers also received jail terms. Five of the older prisoners were sentenced to be hanged.
        Black Hand members were arrested and tried before a Serbian court in Salonika in 1917 on fabricated charges of high treason; 
        the Black Hand was disbanded and three of its leaders were executed. Much of what is known about the assassinations comes from 
        these two trials and related records. Princip's legacy was re-evaluated following the breakup of Yugoslavia, and public 
        opinion of him in the successor states is largely divided along ethnic lines.''',
        '''Sylvia Plath (/plæθ/; October 27, 1932 – February 11, 1963) was an American poet, novelist, and short story writer. 
        She is credited with advancing the genre of confessional poetry and is best known for two of her published collections, 
        The Colossus and Other Poems (1960) and Ariel (1965), as well as The Bell Jar, a semi-autobiographical novel published shortly 
        before her suicide in 1963. The Collected Poems was published in 1981, which included previously unpublished works. 
        For this collection Plath was awarded a Pulitzer Prize in Poetry in 1982, making her the fourth to receive this honour 
        posthumously.[1] Born in Boston, Massachusetts, Plath graduated from Smith College in Massachusetts and the University of 
        Cambridge, England, where she was a student at Newnham College. She married fellow poet Ted Hughes in 1956, and they lived 
        together in the United States and then in England. Their relationship was tumultuous and, in her letters, Plath alleges 
        abuse at his hands.[2] They had two children before separating in 1962.
        Plath was clinically depressed for most of her adult life, and was treated multiple times with electroconvulsive therapy (ECT).[3] 
        She killed herself in 1963.''',
        '''The fall of the Berlin Wall (German: Mauerfall) on 9 November 1989, during the Peaceful Revolution, was a pivotal event 
        in world history which marked the destruction of the Berlin Wall and the figurative Iron Curtain and one of the series of 
        events that started the fall of communism in Central and Eastern Europe, preceded by the Solidarity Movement in Poland. 
        The fall of the inner German border took place shortly afterwards. An end to the Cold War was declared at the Malta Summit 
        three weeks later and the German reunification took place in October the following year.''',
        '''John Fitzgerald Kennedy (May 29, 1917 – November 22, 1963), often referred to by his initials JFK, was an American politician 
        who served as the 35th president of the United States from 1961 until his assassination in 1963. He was the youngest person to 
        assume the presidency by election and the youngest president at the end of his tenure.[2] Kennedy served at the height of the 
        Cold War, and the majority of his foreign policy concerned relations with the Soviet Union and Cuba. A Democrat, Kennedy 
        represented Massachusetts in both houses of the U.S. Congress prior to his presidency.
        Born into the prominent Kennedy family in Brookline, Massachusetts, Kennedy graduated from Harvard University in 1940 before joining 
        the U.S. Naval Reserve the following year. During World War II, he commanded a series of PT boats in the Pacific theater. Kennedy's 
        survival following the sinking of PT-109 and his rescue of his fellow sailors made him a war hero and earned the Navy and Marine 
        Corps Medal, but left him with serious injuries. After a brief stint in journalism, Kennedy represented a working-class Boston 
        district in the U.S. House of Representatives from 1947 to 1953. He was subsequently elected to the U.S. Senate and served as the 
        junior senator for Massachusetts from 1953 to 1960. While in the Senate, Kennedy published his book, Profiles in Courage, which 
        won a Pulitzer Prize. Kennedy ran in the 1960 presidential election. His campaign gained momentum after the first televised 
        presidential debates in American history, and he was elected president, narrowly defeating Republican opponent Richard Nixon, 
        who was the incumbent vice president. He was the first Catholic elected president.
        Kennedy's administration included high tensions with communist states in the Cold War. As a result, he increased the number of 
        American military advisers in South Vietnam. The Strategic Hamlet Program began in Vietnam during his presidency. In April 1961,
        he authorized an attempt to overthrow the Cuban government of Fidel Castro in the failed Bay of Pigs Invasion. In November 1961, 
        he authorized the Operation Mongoose, also aimed at removing the communists from power in Cuba. He rejected Operation Northwoods 
        in March 1962, but his administration continued to plan for an invasion of Cuba in the summer of 1962. The following October, U.S. 
        spy planes discovered Soviet missile bases had been deployed in Cuba; the resulting period of tensions, termed the Cuban Missile Crisis, 
        nearly resulted in the breakout of a global thermonuclear conflict. He also signed the first nuclear weapons treaty in October 1963. 
        Kennedy presided over the establishment of the Peace Corps, Alliance for Progress with Latin America, and the continuation of the 
        Apollo program with the goal of landing a man on the Moon before 1970. He also supported the civil rights movement but was only 
        somewhat successful in passing his New Frontier domestic policies.
        On November 22, 1963, Kennedy was assassinated in Dallas. His vice president, Lyndon B. Johnson, assumed the presidency upon 
        Kennedy's death. Lee Harvey Oswald, a former U.S. Marine, was arrested for the assassination, but he was shot and killed by Jack 
        Ruby two days later. The FBI and the Warren Commission both concluded Oswald had acted alone, but conspiracy theories about the
        assassination still persist. After Kennedy's death, Congress enacted many of his proposals, including the Civil Rights Act of 
        1964 and the Revenue Act of 1964. Kennedy ranks highly in polls of U.S. presidents with historians and the general public. 
        His personal life has also been the focus of considerable sustained interest following public revelations in the 1970s of his 
        chronic health ailments and extramarital affairs. Kennedy is the most recent U.S. president to have died in office.''',
        '''
        The Declaration of Independence, headed The unanimous Declaration of the thirteen united States of America, is the founding document of the United States. It was adopted on July 4, 1776 by the Second Continental Congress meeting at the Pennsylvania State House in Philadelphia, later renamed Independence Hall. The declaration explains to the world why the thirteen colonies regarded themselves as independent sovereign states no longer subject to British colonial rule.

The Declaration of Independence was signed by 56 delegates to the Second Continental Congress, who came to be known as the nation's Founding Fathers. The 56 included delegates from New Hampshire, Massachusetts Bay, Rhode Island and Providence Plantations, Connecticut, New York, New Jersey, Pennsylvania, Maryland, Delaware, Virginia, North Carolina, South Carolina, and Georgia. The declaration became one of the most circulated and widely reprinted documents in early American history.
The Committee of Five drafted the declaration to be ready when Congress voted on independence. John Adams, a leading proponent of independence, persuaded the Committee of Five to charge Thomas Jefferson with writing the document's original draft, which the Second Continental Congress then edited. The declaration was a formal explanation of why the Continental Congress voted to declare American independence from the Kingdom of Great Britain, a year after the American Revolutionary War began in April 1775. The Lee Resolution for independence was passed unanimously by the Congress on July 2, 1776.
After ratifying the text on July 4, Congress issued the Declaration of Independence in several forms. It was initially published as the printed Dunlap broadside that was widely distributed and read to the public. Jefferson's original draft is currently preserved at the Library of Congress in Washington, D.C., complete with changes made by Adams and Benjamin Franklin, and Jefferson's notes of changes made by Congress. The best-known version of the Declaration is the signed copy now displayed at the National Archives in Washington, D.C., which is popularly regarded as the official document. This copy, engrossed by Timothy Matlack, was ordered by Congress on July 19 and signed primarily on August 2, 1776.[2][3]
The declaration justified the independence of the United States by listing 27 colonial grievances against King George III and by asserting certain natural and legal rights, including a right of revolution. Its original purpose was to announce independence, and references to the text of the declaration were few in the following years. Abraham Lincoln made it the centerpiece of his policies and his rhetoric, as in the Gettysburg Address of 1863.[4] Since then, it has become a well-known statement on human rights, particularly its second sentence: "We hold these truths to be self-evident, that all men are created equal, that they are endowed by their Creator with certain unalienable Rights, that among these are Life, Liberty and the pursuit of Happiness."
The declaration was made to guarantee equal rights for every person.[5] Stephen Lucas called it "one of the best-known sentences in the English language",[6] with historian Joseph Ellis writing that the document contains "the most potent and consequential words in American history".[7] The passage came to represent a moral standard to which the United States should strive. This view was notably promoted by Lincoln, who considered the Declaration to be the foundation of his political philosophy and argued that it is a statement of principles through which the United States Constitution should be interpreted.[8]: 126 
The Declaration of Independence inspired many similar documents in other countries, the first being the 1789 Declaration of United Belgian States issued during the Brabant Revolution in the Austrian Netherlands. It also served as the primary model for numerous declarations of independence in Europe, Latin America, Africa (Liberia), and Oceania (New Zealand) during the first half of the 19th century.
        ''',
        '''
        Alaska (/əˈlæskə/ (listen) ə-LAS-kə) is a U.S. state on the northwest extremity of North America. A semi-exclave of the U.S., it borders British Columbia and Yukon in Canada to the east, and it shares a western maritime border in the Bering Strait with the Russian Federation's Chukotka Autonomous Okrug. To the north are the Chukchi and Beaufort Seas of the Arctic Ocean, and the Pacific Ocean lies to the south and southwest.

Alaska is the largest U.S. state by area, comprising more total area than the next three largest states of Texas, California, and Montana combined, and is the seventh-largest subnational division in the world. It is the third-least populous and most sparsely populated U.S. state, but with a population of 736,081 as of 2020, is the continent's most populous territory located mostly north of the 60th parallel, with more than quadruple the combined populations of Northern Canada and Greenland.[3] The state capital of Juneau is the second-largest city in the United States by area, and the former capital of Alaska, Sitka, is the largest U.S. city by area. Approximately half of Alaska's residents live within the Anchorage metropolitan area.
Indigenous people have lived in Alaska for thousands of years, and it is widely believed that the region served as the entry point for the initial settlement of North America by way of the Bering land bridge. The Russian Empire was the first to actively colonize the area beginning in the 18th century, eventually establishing Russian America, which spanned most of the current state, and promoted and maintained a native Alaskan Creole population.[4] The expense and logistical difficulty of maintaining this distant possession prompted its sale to the U.S. in 1867 for US$7.2 million (equivalent to $151 million in 2022). The area went through several administrative changes before becoming organized as a territory on May 11, 1912. It was admitted as the 49th state of the U.S. on January 3, 1959.[5]
Abundant natural resources have enabled Alaska—with one of the smallest state economies—to have one of the highest per capita incomes, with commercial fishing, and the extraction of natural gas and oil, dominating Alaska's economy. U.S. Armed Forces bases and tourism also contribute to the economy; more than half the state is federally-owned land containing national forests, national parks, and wildlife refuges.
The Indigenous population of Alaska is proportionally the highest of any U.S. state, at over 15 percent.[6] Various Indigenous languages are spoken, and Alaskan Natives are influential in local and state politics.
        ''',
        '''
        The Industrial Revolution (also known as the First Industrial Revolution) was a period of global transition of human economy towards more efficient and stable manufacturing processes that succeeded the Agricultural Revolution, starting from Great Britain, continental Europe, and the United States, that occurred during the period from around 1760 to about 1820–1840.[1] This transition included going from hand production methods to machines; new chemical manufacturing and iron production processes; the increasing use of water power and steam power; the development of machine tools; and the rise of the mechanized factory system. Output greatly increased, and a result was an unprecedented rise in population and in the rate of population growth. The textile industry was the first to use modern production methods,[2]: 40  and textiles became the dominant industry in terms of employment, value of output, and capital invested.

On a structural level the Industrial Revolution asked society the so-called social question, demanding new ideas for managing large groups of individuals. Growing poverty on one hand and growing population and materialistic wealth on the other caused tensions between the very rich and the poorest people within society.[3] These tensions were sometimes violently released[4] and led to philosophical ideas such as socialism, communism and anarchism.

The Industrial Revolution began in Great Britain, and many of the technological and architectural innovations were of British origin.[5][6] By the mid-18th century, Britain was the world's leading commercial nation,[7] controlling a global trading empire with colonies in North America and the Caribbean. Britain had major military and political hegemony on the Indian subcontinent; particularly with the proto-industrialised Mughal Bengal, through the activities of the East India Company.[8][9][10][11] The development of trade and the rise of business were among the major causes of the Industrial Revolution.[2]: 15 
The Industrial Revolution marked a major turning point in history. Comparable only to humanity's adoption of agriculture with respect to material advancement,[12] the Industrial Revolution influenced in some way almost every aspect of daily life. In particular, average income and population began to exhibit unprecedented sustained growth. Some economists have said the most important effect of the Industrial Revolution was that the standard of living for the general population in the Western world began to increase consistently for the first time in history, although others have said that it did not begin to improve meaningfully until the late 19th and 20th centuries.[13][14][15] GDP per capita was broadly stable before the Industrial Revolution and the emergence of the modern capitalist economy,[16] while the Industrial Revolution began an era of per-capita economic growth in capitalist economies.[17] Economic historians agree that the onset of the Industrial Revolution is the most important event in human history since the domestication of animals and plants.[18]
The precise start and end of the Industrial Revolution is still debated among historians, as is the pace of economic and social changes.[19][20][21][22] Eric Hobsbawm held that the Industrial Revolution began in Britain in the 1780s and was not fully felt until the 1830s or 1840s,[19] while T. S. Ashton held that it occurred roughly between 1760 and 1830.[20] Rapid industrialisation first began in Britain, starting with mechanized textiles spinning in the 1780s,[23] with high rates of growth in steam power and iron production occurring after 1800. Mechanized textile production spread from Great Britain to continental Europe and the United States in the early 19th century, with important centres of textiles, iron and coal emerging in Belgium and the United States and later textiles in France.[2]
An economic recession occurred from the late 1830s to the early 1840s when the adoption of the Industrial Revolution's early innovations, such as mechanized spinning and weaving, slowed and their markets matured. Innovations developed late in the period, such as the increasing adoption of locomotives, steamboats and steamships, and hot blast iron smelting. New technologies such as the electrical telegraph, widely introduced in the 1840s and 1850s, were not powerful enough to drive high rates of growth. Rapid economic growth began to occur after 1870, springing from a new group of innovations in what has been called the Second Industrial Revolution. These innovations included new steel-making processes, mass production, assembly lines, electrical grid systems, the large-scale manufacture of machine tools, and the use of increasingly advanced machinery in steam-powered factories.
        ''',
        '''
        The Nineteenth Amendment (Amendment XIX) to the United States Constitution prohibits the United States and its states from denying the right to vote to citizens of the United States on the basis of sex, in effect recognizing the right of women to a vote. The amendment was the culmination of a decades-long movement for women's suffrage in the United States, at both the state and national levels, and was part of the worldwide movement towards women's suffrage and part of the wider women's rights movement. The first women's suffrage amendment was introduced in Congress in 1878. However, a suffrage amendment did not pass the House of Representatives until May 21, 1919, which was quickly followed by the Senate, on June 4, 1919. It was then submitted to the states for ratification, achieving the requisite 36 ratifications to secure adoption, and thereby go into effect, on August 18, 1920. The Nineteenth Amendment's adoption was certified on August 26, 1920.
Before 1776, women had a vote in several of the colonies in what would become the United States, but by 1807 every state constitution had denied women even limited suffrage. Organizations supporting women's rights became more active in the mid-19th century and, in 1848, the Seneca Falls convention adopted the Declaration of Sentiments, which called for equality between the sexes and included a resolution urging women to secure the vote. Pro-suffrage organizations used a variety of tactics including legal arguments that relied on existing amendments. After those arguments were struck down by the U.S. Supreme Court, suffrage organizations, with activists like Susan B. Anthony and Elizabeth Cady Stanton, called for a new constitutional amendment guaranteeing women the same right to vote possessed by men.
By the late 19th century, new states and territories, particularly in the West, began to grant women the right to vote. In 1878, a suffrage proposal that would eventually become the Nineteenth Amendment was introduced to Congress, but was rejected in 1887. In the 1890s, suffrage organizations focused on a national amendment while still working at state and local levels. Lucy Burns and Alice Paul emerged as important leaders whose different strategies helped move the Nineteenth Amendment forward. Entry of the United States into World War I helped to shift public perception of women's suffrage. The National American Woman Suffrage Association, led by Carrie Chapman Catt, supported the war effort, making the case that women should be rewarded with enfranchisement for their patriotic wartime service. The National Woman's Party staged marches, demonstrations, and hunger strikes while pointing out the contradictions of fighting abroad for democracy while limiting it at home by denying women the right to vote. The work of both organizations swayed public opinion, prompting President Woodrow Wilson to announce his support of the suffrage amendment in 1918. It passed in 1919 and was adopted in 1920, withstanding two legal challenges, Leser v. Garnett and Fairchild v. Hughes.
The Nineteenth Amendment enfranchised 26 million American women in time for the 1920 U.S. presidential election, but the powerful women's voting bloc that many politicians feared failed to fully materialize until decades later. Additionally, the Nineteenth Amendment failed to fully enfranchise African American, Asian American, Hispanic American, and Native American women (see § Limitations). Shortly after the amendment's adoption, Alice Paul and the National Woman's Party began work on the Equal Rights Amendment, which they believed was a necessary additional step towards equality.
        ''',
        '''
        Madam C. J. Walker (born Sarah Breedlove; December 23, 1867 – May 25, 1919) was an African American entrepreneur, philanthropist, and political and social activist. She is recorded as the first female self-made millionaire in America in the Guinness Book of World Records.[1] Multiple sources mention that although other women (like Mary Ellen Pleasant) might have been the first, their wealth is not as well-documented.[1][2][3]
Walker made her fortune by developing and marketing a line of cosmetics and hair care products for black women through the business she founded, Madam C. J. Walker Manufacturing Company. She became known also for her philanthropy and activism. She made financial donations to numerous organizations such as the NAACP, and became a patron of the arts. Villa Lewaro, Walker's lavish estate in Irvington, New York, served as a social gathering place for the African-American community. At the time of her death, she was considered the wealthiest African-American businesswoman and wealthiest self-made black woman in America.[4] Her name was a version of "Mrs. Charles Joseph Walker", after her third husband.
        '''
    ]

    qa = BERT_QA()
    questions = ["When was the treaty of versiallies signed for world war one?",
                 "What year was Archduke Franz Ferdinand assassinated?",
                 "What was Sylvia Plath's cause of death?", "When did the Berlin wall fall?",
                 "Where was JFK assassinated?", "When was the declaration of independence signed?",
                 "the united states bought alaska from which country?",
                 "Which era marked a switch from agricultural to industrial practices?",
                 "Fill in the blake: the 19th amendment guarantees _ the right to vote",
                 "Who was the first woman to make a million dollars in the United States?"]

    cycle_range = random.randrange(1, 5, 2)

    for i in range(cycle_range):
        for i in range(len(questions)):
            qa.get_split_tokens(contexts[i])
            qa.pre_process(questions[i], qa.split_total[0])
            print(f"Question: {questions[i]}")
            qa.predict()
            print(f"Answer: {qa.answer_list[i]}\n")



main()
