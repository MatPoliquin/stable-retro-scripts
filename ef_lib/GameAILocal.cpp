
#include "GameAI.h"
#include "nhl94.h"
#include "DefaultGameAI.h"

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
  std::cout << "CreateGameAI" << std::endl;
  std::cout << name << std::endl;

  std::filesystem::path path = name;

  std::string game_name = path.parent_path().filename();
  std::cout << game_name << std::endl;

  //Test_Resnet();

  GameAILocal * ptr = nullptr;

  if(game_name == "NHL941on1-Genesis")
  {
    std::cout << "GAME SUPPORTED!" << std::endl;
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
  }

  
  return (GameAI *) ptr;
}

//=======================================================
// RetroModelPytorch::Forward
//=======================================================
RetroModel * GameAILocal::LoadModel(const char * path)
{
    RetroModelPytorch * model = new RetroModelPytorch();

    model->LoadModel(std::string(path));

    return dynamic_cast<RetroModel*>(model);
}

//=======================================================
// RetroModelPytorch::Forward
//=======================================================
void GameAILocal::DebugPrint(const char * msg)
{
    if (showDebug && debugLogFunc)
    {
        std::cout << msg << std::endl;

        debugLogFunc(0, msg);
    }
}


