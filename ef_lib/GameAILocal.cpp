
#include <stdexcept>

#include "GameAI.h"
#include "./games/NHL94GameAI.h"
#include "./games/DefaultGameAI.h"
#include "./utils/json.hpp"

#if _WIN32
#define DllExport   __declspec( dllexport )
#else
#define DllExport
#endif


//=======================================================
// CreateGameAI
//=======================================================
extern "C"  DllExport GameAI * CreateGameAI(const char * name)
{
    std::filesystem::path path = name;
    std::string game_name = path.parent_path().filename();
    
    GameAILocal * ptr = nullptr;

    if(game_name == "NHL941on1-Genesis")
    {
      ptr = new NHL94GameAI();
    }
    else
    {
      ptr = new DefaultGameAI();
    }

    if (ptr)
    {
      ptr->full_path = path.string();
      ptr->dir_path = path.parent_path().string();
      ptr->game_name = game_name;

      ptr->DebugPrint("CreateGameAI");
      ptr->DebugPrint(name);
      ptr->DebugPrint(game_name.c_str());
    }

  
  return (GameAI *) ptr;
}

//=======================================================
// GameAILocal::InitRAM
//=======================================================
void GameAILocal::InitRAM(void * ram_ptr, int ram_size)
{
    std::filesystem::path memDataPath = dir_path;
    memDataPath += "/data.json";

    //retro_data.load()
    //std::cout << memDataPath << std::endl;
    retro_data.load(memDataPath.string());
    
    Retro::AddressSpace* m_addressSpace = nullptr;
    m_addressSpace = &retro_data.addressSpace();
	  m_addressSpace->reset();
	  //Retro::configureData(data, m_core);
	  //reconfigureAddressSpace();
    retro_data.addressSpace().setOverlay(Retro::MemoryOverlay{ '=', '>', 2 });

	  m_addressSpace->addBlock(16711680, ram_size, ram_ptr);
    std::cout << "RAM size:" << ram_size << std::endl;
    std::cout << "RAM ptr:" << ram_ptr << std::endl;
}

//=======================================================
// GameAILocal::LoadConfig
//=======================================================
void GameAILocal::LoadConfig()
{
  DebugPrint("GameAILocal::LoadConfig()");

  std::filesystem::path configPath = dir_path;
  configPath += "/config.json";
  DebugPrint(configPath.c_str());

  std::ifstream file;
  try {
	  file.open(configPath);
    //std::cout << file.rdbuf();
    //std::cerr << "Error: " << strerror(errno);
    //std::cout << file.get();
  }
  catch (std::exception & e){
    DebugPrint("Error opening config file");
    DebugPrint(e.what());
  }

  //file.clear();
  //file.seekg(0, std::ios::beg);

  using nlohmann::json;

	json manifest;
	try {
		file >> manifest;
	} catch (json::exception& e) {
		DebugPrint("Error Loading config");
    DebugPrint(e.what());
    return;
	}

	const auto& p1 = const_cast<const json&>(manifest).find("p1");
	if (p1 == manifest.cend())
  {
		DebugPrint("Error Loading config, no p1");
    return;
	}

  for (auto var = p1->cbegin(); var != p1->cend(); ++var)
  {
    if(var.key() == "models")
    {
      for (auto model = var.value().cbegin(); model != var.value().cend(); ++model)
      {
        std::filesystem::path modelPath = dir_path;
        modelPath += "/";
        modelPath += model.value().get<std::string>();

        RetroModel * load_model = this->LoadModel(modelPath.string().c_str());

        models.insert(std::pair<std::string, RetroModel*>(model.key(), load_model));
      }
    }
  }
}

//=======================================================
// GameAILocal::LoadModel
//=======================================================
RetroModel * GameAILocal::LoadModel(const char * path)
{
    RetroModelPytorch * model = new RetroModelPytorch();

    model->LoadModel(std::string(path));

    return dynamic_cast<RetroModel*>(model);
}

//=======================================================
// GameAILocal::DebugPrint
//=======================================================
void GameAILocal::DebugPrint(const char * msg)
{
    std::cout << msg << std::endl;
    if (showDebug && debugLogFunc)
    {
        std::cout << msg << std::endl;

        debugLogFunc(0, msg);
    }
}


