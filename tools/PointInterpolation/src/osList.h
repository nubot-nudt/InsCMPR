#ifndef _OSLIST_H
#define _OSLIST_H

#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>

#include <boost/filesystem.hpp>

std::string removeOblique(std::string s)
{
    if(s.length()==0)
    {
        return s;
    }
    for(int i=s.length()-1;i>=0;i--)
    {
        if(s[i]=='\\' || s[i]=='/')
        {
            continue;
        }
        else
        {
            s = s.substr(0, i+1);
            break;
        }
    }
    return s;
}

std::string join(std::string a, std::string b)
{
    a = removeOblique(a);
    b = removeOblique(b);
    return a+"/"+b;
}

std::string join(std::string a, std::string b, std::string c)
{
    a = removeOblique(a);
    b = removeOblique(b);
    c = removeOblique(c);
    return a+"/"+b+"/"+c;
}

static bool cmp(std::string a, std::string b)
{
	if (a < b)
	{
		return true;
	}
	return false;
}

std::vector<std::string> pathList(std::string path, bool sort=false)
{
    std::vector<std::string> files;
    if(!boost::filesystem::exists(path))
    {
        std::cout<<"dont exist"<<std::endl;
        return files;
    }
    if(!boost::filesystem::is_directory(path))
    {
        std::cout<<"not a dir"<<std::endl;
        return files;
    }
    struct dirent* ptr;
    DIR* dir;
    dir = opendir(path.c_str());

    while((ptr=readdir(dir)) != NULL)
    {
        if(ptr->d_name[0] == '.')
            continue;
        files.push_back(ptr->d_name);
    }

    if(sort)
    {
        std::sort(files.begin(), files.end(), cmp);
    }
    
    return files;
}

// from "1.23 1.45 456" to [1.23, 1.45, 456]
std::vector<double> segment(std::string words, char delimiter=' ')
{
    std::vector<double> nums;
    std::string word;
    for(int i=0; i<words.length(); i++)
    {
        if(words[i]==delimiter)
        {
            nums.push_back(std::stof(word));
            word="";
        }
        else
        {
            word = word + words[i];
        }
    }
    nums.push_back(std::stof(word));
    return nums;
}

bool checkPostfix(std::string fileName, std::string postfix)
{
    if(fileName.length()<=3)
    {
        return false;
    }
    std::string _postfix = "";
    for(int i=fileName.length()-1; i>0; i--)
    {
        if(fileName[i]!='.')
        {
            _postfix += fileName[i];
        }
        else
        {
            if(_postfix==postfix)
            {
                return true;
            }
            break;
        }
    }
    return false;
}

#endif