def read(levelName, read = True, count = 0, entities = []):
    level = []
    try:
        if levelName.endswith(".txt"):
            f = open(levelName,"r")
        else:
            f = open(levelName +".txt","r")
        levelStr = str(f.read())
        # levelList = levelStr.split("\n")
        # for i in levelList: print(i.split(","))
        if len(levelStr) > 10:
            while read:
                #Find the Name
                for i in levelStr:
                    if i == "{":
                        break
                    else:
                        levelName = levelName + i
                    count += 1

                #Enter the basic level bracket
                count += 2

                #check for metadata
                while levelStr[count] != "}":
                    if levelStr[count] == '"': #!!! check if file is empty
                        metadata = ""
                        count += 1
                        while levelStr[count] !=  '"':
                            metadata += levelStr[count]
                            count += 1
                        level.append(metadata)

                    elif levelStr[count] == "[":
                        count += 1
                        level.append([0,0,0,0,"",0,0])

                        #find the coordinates
                        for i in range(4):
                            num = ""
                            while levelStr[count+i] != ",":
                                num += levelStr[count+i]
                                count += 1

                            if len(num) > 0:
                                level[-1][i] = float(num)
                        count += 4

                        #find the texture
                        texture = ""
                        while levelStr[count] != ",":
                            texture += levelStr[count]
                            count += 1

                        level[-1][4] = texture
                        count += 1
                        #check for optional addition (mode, damage)
                        if levelStr[count] == "]":
                            pass
                        elif levelStr[count] == "}":
                            count -= 1
                        elif levelStr[count].isdigit(): #check for texture mode
                            num = ""
                            while levelStr[count].isdigit():
                                num += levelStr[count]
                                count += 1
                            level[len(level)-1][5] = int(num)

                            count += 1

                            if levelStr[count] == "]":
                                pass
                            elif levelStr[count] == "}":
                                count -= 1
                            elif levelStr[count].isdigit() or levelStr[count] == "-": #check for damage
                                num = ""
                                while levelStr[count].isdigit() or levelStr[count] == "-":
                                    num += levelStr[count]
                                    count += 1
                                level[len(level)-1][6] = int(num)

                    count +=1
                read = False
            return level, entities
        else:
            return [], []
    except Exception as e:
        print(e)
        return 1
