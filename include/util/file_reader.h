#pragma once

#include <fstream>

namespace dvo
{
namespace util
{
template <class EntryT>
class FileReader
{
public:
    FileReader(std::string &file) : hasEntry_(false),
                                    file_(file),
                                    file_stream_(file.c_str())
    {
    }

    virtual ~FileReader()
    {
        file_stream_.close();
    }

    void skip(int num_lines)
    {
        for (int idx = 0; idx < num_lines; ++idx)
        {
            if (!file_stream_.good())
                continue;

            file_stream_.ignore(1024, '\n');
            assert(file_stream_.gcount() < 1024);
        }
    }

    void skipComments()
    {
        while (file_stream_.good() && file_stream_.peek() == '#')
        {
            skip(1);
        }
    }

    /**
   * Moves to the next entry in the file. Returns true, if there was a next entry, false otherwise.
   */
    bool next()
    {
        if (file_stream_.good() && !file_stream_.eof())
        {
            file_stream_ >> entry_;
            hasEntry_ = true;

            return true;
        }

        return false;
    }

    /**
   * Read all entries at once.
   */
    void readAllEntries(std::vector<EntryT> &entries)
    {
        if (!hasEntry_)
            next();

        do
        {
            entries.push_back(entry_);
        } while (next());
    }

private:
    bool hasEntry_;
    EntryT entry_;

    std::string file_;
    std::ifstream file_stream_;
};

class Groundtruth
{
public:
    double timestamp_;

    double postion_x_;
    double postion_y_;
    double postion_z_;

    double orientation_x_;
    double orientation_y_;
    double orientation_z_;
    double orientation_w_;

    friend std::ostream &operator<<(std::ostream &out, const Groundtruth &pair);
    friend std::istream &operator>>(std::istream &in, Groundtruth &pair);
};

class RGBDPair
{
public:
    double rgb_timestamp_;
    std::string rgb_file_;

    double depth_timestamp_;
    std::string depth_file_;

    friend std::ostream &operator<<(std::ostream &out, const RGBDPair &pair);
    friend std::istream &operator>>(std::istream &in, RGBDPair &pair);
};

std::ostream &operator<<(std::ostream &out, const Groundtruth &gt)
{
    out
        << gt.timestamp_ << " "
        << gt.postion_x_ << " "
        << gt.postion_y_ << " "
        << gt.postion_z_ << " "
        << gt.orientation_x_ << " "
        << gt.orientation_y_ << " "
        << gt.orientation_z_ << " "
        << gt.orientation_w_ << std::endl;

    return out;
}

std::istream &operator>>(std::istream &in, Groundtruth &gt)
{

    in >> gt.timestamp_;
    in >> gt.postion_x_;
    in >> gt.postion_y_;
    in >> gt.postion_z_;
    in >> gt.orientation_x_;
    in >> gt.orientation_y_;
    in >> gt.orientation_z_;
    in >> gt.orientation_w_;

    return in;
}

std::ostream &operator<<(std::ostream &out, const RGBDPair &pair)
{
    out
        << pair.rgb_timestamp_ << " "
        << pair.rgb_file_ << " "
        << pair.depth_timestamp_ << " "
        << pair.depth_file_ << std::endl;

    return out;
}

std::istream &operator>>(std::istream &in, RGBDPair &pair)
{
    in >> pair.rgb_timestamp_;
    in >> pair.rgb_file_;
    in >> pair.depth_timestamp_;
    in >> pair.depth_file_;

    return in;
}

} // namespace util
} // namespace dvo